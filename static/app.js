const fileEl = document.getElementById('file');
const btn = document.getElementById('btn');
const img = document.getElementById('img');
const out = document.getElementById('out');

let file = null;
fileEl.addEventListener('change', e => {
  file = e.target.files[0] || null;
  btn.disabled = !file;
  if (file) {
    const reader = new FileReader();
    reader.onload = ev => img.src = ev.target.result;
    reader.readAsDataURL(file);
  } else img.src = '';
});

btn.addEventListener('click', async () => {
  if (!file) return;
  out.textContent = 'Predicting...';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const lines = [
      `Prediction: ${data.category} (${(data.confidence*100).toFixed(2)}%)`,
      'Top-5:'
    ];
    data.top5.forEach((t,i)=>lines.push(`${i+1}. ${t[0]} ${(t[1]*100).toFixed(2)}%`));
    out.textContent = lines.join('\n');
  } catch (err) {
    out.textContent = 'Error: ' + err;
  }

});
