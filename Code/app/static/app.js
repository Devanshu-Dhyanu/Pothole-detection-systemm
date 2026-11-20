const cameraInput = document.getElementById('cameraInput');
const previewCard = document.getElementById('previewCard');
const imagePreview = document.getElementById('imagePreview');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultCard = document.getElementById('resultCard');

const priorityBadge = document.getElementById('priorityBadge');
const defectTitle = document.getElementById('defectTitle');
const statPotholes = document.getElementById('statPotholes');
const statPsi = document.getElementById('statPsi');
const statDepth = document.getElementById('statDepth');
const actionText = document.getElementById('actionText');
const costText = document.getElementById('costText');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const cameraCard = document.querySelector('.camera-card');

cameraInput.addEventListener('change', (e) => analyzeFile(e.target.files[0]));

['dragenter', 'dragover'].forEach((eventName) => {
  cameraCard.addEventListener(eventName, (e) => {
    e.preventDefault();
    cameraCard.classList.add('dragging');
  });
});
['dragleave', 'drop'].forEach((eventName) => {
  cameraCard.addEventListener(eventName, () => cameraCard.classList.remove('dragging'));
});
cameraCard.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) analyzeFile(file);
});

async function analyzeFile(file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    imagePreview.src = event.target.result;
    previewCard.classList.add('has-image');
    previewPlaceholder.style.display = 'none';
    loadingSpinner.classList.add('visible');
    resultCard.classList.add('loading');
  };
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/api/v1/detect', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error('Analysis request failed');
    const data = await response.json();
    loadingSpinner.classList.remove('visible');
    resultCard.classList.remove('loading');

    if (data.success) {
      const sev = data.severity_assessment;
      priorityBadge.innerText = sev.priority_level;
      if (sev.priority_level.includes('P1')) {
        priorityBadge.style.background = '#dc2626';
        priorityBadge.style.color = '#ffffff';
      } else if (sev.priority_level.includes('P2')) {
        priorityBadge.style.background = '#f59e0b';
        priorityBadge.style.color = '#1b1202';
      } else {
        priorityBadge.style.background = '#16a34a';
        priorityBadge.style.color = '#ffffff';
      }

      defectTitle.innerText = `${sev.hazard_class}`;
      statPotholes.innerText = sev.num_potholes;
      statPsi.innerText = `${sev.psi}/100`;
      statDepth.innerText = `${sev.max_depth_cm} cm`;
      actionText.innerText = `Action: ${sev.action_advisory}`;
      costText.innerText = `Estimated Repair: ₹ ${sev.estimated_repair_cost_inr.toLocaleString()} INR`;
    }
  } catch (err) {
    loadingSpinner.classList.remove('visible');
    resultCard.classList.remove('loading');
    defectTitle.innerText = 'Analysis unavailable';
    actionText.innerText = 'Detection server is offline or the image could not be processed.';
    costText.innerText = 'Please try again.';
  }
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js').catch(() => {});
}
