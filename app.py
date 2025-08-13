from pathlib import Path
from typing import List, Tuple
import io

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# =============================
# Config
# =============================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"          # put your .pt / .pth here
STATIC_DIR = BASE_DIR / "static"          # index.html, app.js, styles.css

SVHN_MEAN = (0.438, 0.444, 0.473)
SVHN_STD  = (0.198, 0.201, 0.197)
LABEL_NAMES = [str(i) for i in range(10)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Optional model definition (used if we load state_dict weights)
# =============================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.short = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.short = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.short(x)
        return F.relu(out)

class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

def ResNet18_CIFAR(num_classes=10):
    return ResNetCIFAR(BasicBlock, [2,2,2,2], num_classes)

# =============================
# Load model (TorchScript preferred; fallback to state_dict)
# =============================
script_path = MODELS_DIR / "svhn_resnet18_scripted.pt"
weights_path = MODELS_DIR / "svhn_resnet18_best_weights.pth"

model_script = None
model = None

if script_path.exists():
    model_script = torch.jit.load(str(script_path), map_location=device)
    model_script.eval()
    print("Loaded TorchScript:", script_path)
elif weights_path.exists():
    model = ResNet18_CIFAR(10).to(device)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print("Loaded state_dict weights:", weights_path)
else:
    raise FileNotFoundError("Place model in 'models/': svhn_resnet18_scripted.pt or svhn_resnet18_best_weights.pth")

# =============================
# App + middleware
# =============================
app = FastAPI(title="SVHN Digit Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =============================
# Inference helpers
# =============================

def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((32, 32))
    x = torch.from_numpy(np.array(img)).float() / 255.0   # HWC
    x = x.permute(2, 0, 1)                                # CHW
    mean = torch.tensor(SVHN_MEAN).view(3,1,1)
    std  = torch.tensor(SVHN_STD).view(3,1,1)
    x = (x - mean) / std
    return x.unsqueeze(0).to(device)                      # NCHW

@torch.inference_mode()
def infer(x: torch.Tensor):
    if model_script is not None:
        logits = model_script(x)
    else:
        logits = model(x)
    probs = logits.softmax(dim=1).squeeze(0)
    top_p, top_i = torch.topk(probs, k=5)
    top_p = top_p.cpu().numpy().tolist()
    top_i = top_i.cpu().numpy().tolist()
    top5 = [(LABEL_NAMES[i], float(p)) for i, p in zip(top_i, top_p)]
    pred_idx = top_i[0]
    pred_name = LABEL_NAMES[pred_idx]
    conf = float(top_p[0])
    return pred_idx, pred_name, conf, top5

# =============================
# API routes
# =============================
@app.get("/health")
def health():
    return {"ok": True, "device": str(device)}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    idx, name, conf, top5 = infer(preprocess(img))
    return {"index": idx, "category": name, "confidence": conf, "top5": top5}

# =============================
# Mount static frontend *AFTER* routes so it doesn't swallow /api/*
# =============================
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# To run:
# uvicorn app:app --host 127.0.0.1 --port 8000 --reload
