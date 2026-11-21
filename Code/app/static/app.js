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
const historyList = document.getElementById('historyList');
const historyCount = document.getElementById('historyCount');
const historyMessage = document.getElementById('historyMessage');
const mapMessage = document.getElementById('mapMessage');
const mapCount = document.getElementById('mapCount');
const mapEmpty = document.getElementById('mapEmpty');
const exportCsvButton = document.getElementById('exportCsvButton');
const exportPdfButton = document.getElementById('exportPdfButton');
const uploadMessage = document.getElementById('uploadMessage');
const uploadProgressText = document.getElementById('uploadProgressText');
const uploadProgressPercent = document.getElementById('uploadProgressPercent');
const uploadProgressTrack = document.getElementById('uploadProgressTrack');
const uploadProgressBar = document.getElementById('uploadProgressBar');
const systemStatus = document.getElementById('systemStatus');
const systemStatusText = document.getElementById('systemStatusText');
const authStatusText = document.getElementById('authStatusText');
const authScreen = document.getElementById('authScreen');
const authForm = document.getElementById('authForm');
const authError = document.getElementById('authError');
const authTitle = document.getElementById('authTitle');
const authSubmit = document.getElementById('authSubmit');
const nameField = document.getElementById('nameField');
const googleAuthButton = document.getElementById('googleAuthButton');
const googleAuthNote = document.getElementById('googleAuthNote');
let authMode = 'register';
const logoutButton = document.getElementById('logoutButton');
const historyDatabaseName = 'roadguard-history';
const historyStoreName = 'scans';
const maxHistoryEntries = 20;
const maxUploadBytes = 10 * 1024 * 1024;
const allowedImageTypes = new Map([
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.png', 'image/png'],
  ['.webp', 'image/webp']
]);
let scanMap = null;
let scanMarkerLayer = null;
let currentScans = [];

async function updateSystemStatus() {
  try {
    const response = await fetch('/api/v1/health', { cache: 'no-store' });
    if (!response.ok) throw new Error('Health check failed');
    systemStatus.classList.add('online');
    systemStatus.classList.remove('offline');
    systemStatusText.textContent = 'Service available';
  } catch {
    systemStatus.classList.add('offline');
    systemStatus.classList.remove('online');
    systemStatusText.textContent = 'Service unavailable';
  }
}

async function updateAuthStatus() {
  try {
    const response = await fetch('/auth/me', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Account request failed (${response.status})`);
    const auth = await response.json();
    const user = auth.user;

    authScreen.hidden = Boolean(auth.authenticated);
    googleAuthButton.disabled = !auth.configured;
    googleAuthNote.hidden = auth.configured || Boolean(auth.authenticated);
    logoutButton.hidden = !auth.authenticated;

    if (auth.authenticated && user) {
      authStatusText.textContent = `Signed in as ${user.name || user.email || 'Google user'}`;
    } else {
      authStatusText.textContent = 'Signed out';
    }
  } catch (error) {
    authStatusText.textContent = 'Sign-in status unavailable';
    authScreen.hidden = false;
    logoutButton.hidden = true;
    console.error('Unable to load Google sign-in status:', error);
  }
}

document.querySelectorAll('.auth-tab').forEach((tab) => tab.addEventListener('click', () => {
  authMode = tab.dataset.mode;
  document.querySelectorAll('.auth-tab').forEach((item) => item.classList.toggle('active', item === tab));
  const registering = authMode === 'register';
  nameField.hidden = !registering;
  nameField.querySelector('input').required = registering;
  authForm.elements.password.autocomplete = registering ? 'new-password' : 'current-password';
  authForm.elements.password.placeholder = registering ? 'At least 8 characters' : 'Your password';
  authTitle.textContent = registering ? 'Create your account' : 'Welcome back';
  authSubmit.innerHTML = registering ? 'Create account <span>→</span>' : 'Sign in <span>→</span>';
  authError.textContent = '';
}));

authForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  authError.textContent = '';
  authSubmit.disabled = true;
  try {
    const fields = Object.fromEntries(new FormData(authForm));
    const response = await fetch(`/auth/${authMode}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(fields) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Unable to sign in.');
    authForm.reset();
    await updateAuthStatus();
  } catch (error) {
    authError.textContent = error.message;
  } finally {
    authSubmit.disabled = false;
  }
});

googleAuthButton.addEventListener('click', () => {
  if (!googleAuthButton.disabled) window.location.assign('/auth/google/login');
});

logoutButton.addEventListener('click', async () => {
  logoutButton.disabled = true;
  try {
    const response = await fetch('/auth/logout', { method: 'POST' });
    if (!response.ok) throw new Error(`Sign-out failed (${response.status})`);
    await updateAuthStatus();
  } catch (error) {
    authStatusText.textContent = 'Sign-out failed';
    console.error('Unable to sign out:', error);
  } finally {
    logoutButton.disabled = false;
  }
});

function initializeScanMap() {
  if (!window.L) {
    mapMessage.textContent = 'The interactive map could not load. Check your internet connection and refresh.';
    mapEmpty.textContent = 'Map unavailable.';
    return;
  }

  scanMap = window.L.map('scanMap', { scrollWheelZoom: false }).setView([20, 0], 2);
  window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
  }).addTo(scanMap);
  scanMarkerLayer = window.L.layerGroup().addTo(scanMap);
}

function isValidLocation(location) {
  return location
    && Number.isFinite(location.latitude)
    && Number.isFinite(location.longitude)
    && Math.abs(location.latitude) <= 90
    && Math.abs(location.longitude) <= 180;
}

function createMapPopup(scan) {
  const popup = document.createElement('div');
  const title = document.createElement('strong');
  title.textContent = scan.filename;
  const details = document.createElement('p');
  details.textContent = `${scan.priority} · ${scan.hazardClass} · PSI ${scan.psi}/100`;
  const date = document.createElement('span');
  date.textContent = new Date(scan.createdAt).toLocaleString();
  popup.append(title, details, date);
  return popup;
}

function updateScanMap(scans) {
  if (!scanMap || !scanMarkerLayer) return;

  const locatedScans = scans.filter((scan) => isValidLocation(scan.location));
  const invalidLocationCount = scans.filter((scan) => scan.location && !isValidLocation(scan.location)).length;
  mapCount.textContent = `${locatedScans.length} mapped`;
  mapEmpty.hidden = locatedScans.length > 0;
  scanMap.closePopup();
  scanMarkerLayer.clearLayers();

  const markers = locatedScans.map((scan) => {
    const priority = String(scan.priority || '').toUpperCase();
    const markerClass = priority.includes('P1') ? 'map-pin-p1'
      : priority.includes('P2') ? 'map-pin-p2'
        : 'map-pin-p3';
    const icon = window.L.divIcon({
      className: 'roadguard-div-icon',
      html: `<span class="map-pin ${markerClass}"></span>`,
      iconSize: [26, 34],
      iconAnchor: [13, 30],
      popupAnchor: [0, -28]
    });
    const marker = window.L.marker(
      [scan.location.latitude, scan.location.longitude],
      { icon }
    ).bindPopup(createMapPopup(scan));
    scanMarkerLayer.addLayer(marker);
    return marker;
  });

  if (markers.length > 1) {
    scanMap.fitBounds(window.L.featureGroup(markers).getBounds().pad(0.18), { maxZoom: 13 });
  } else if (markers.length === 1) {
    scanMap.setView(markers[0].getLatLng(), 13);
  } else {
    scanMap.setView([20, 0], 2);
  }

  mapMessage.textContent = invalidLocationCount
    ? `${locatedScans.length} scan${locatedScans.length === 1 ? '' : 's'} mapped · ${invalidLocationCount} invalid location${invalidLocationCount === 1 ? '' : 's'} skipped.`
    : locatedScans.length
      ? `${locatedScans.length} scan${locatedScans.length === 1 ? '' : 's'} with saved GPS location${locatedScans.length === 1 ? '' : 's'}.`
      : 'Add a location to a scan to see it on the map.';
  requestAnimationFrame(() => scanMap.invalidateSize());
}

function openHistoryDatabase() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) {
      reject(new Error('Scan history is not supported by this browser.'));
      return;
    }

    const request = indexedDB.open(historyDatabaseName, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(historyStoreName)) {
        database.createObjectStore(historyStoreName, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Unable to open scan history.'));
    request.onblocked = () => reject(new Error('Scan history is blocked by another open tab.'));
  });
}

function getHistoryScans() {
  return openHistoryDatabase().then((database) => new Promise((resolve, reject) => {
    const transaction = database.transaction(historyStoreName, 'readonly');
    const request = transaction.objectStore(historyStoreName).getAll();
    request.onsuccess = () => {
      resolve(request.result.sort((a, b) => b.createdAt.localeCompare(a.createdAt)));
    };
    request.onerror = () => reject(request.error || new Error('Unable to load scan history.'));
    transaction.onabort = () => reject(transaction.error || new Error('Unable to load scan history.'));
  }));
}

function saveHistoryScan(scan) {
  return openHistoryDatabase().then((database) => new Promise((resolve, reject) => {
    const transaction = database.transaction(historyStoreName, 'readwrite');
    const store = transaction.objectStore(historyStoreName);
    const request = store.getAll();

    request.onsuccess = () => {
      const oldestScans = request.result
        .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
        .slice(maxHistoryEntries - 1);
      store.put(scan);
      oldestScans.forEach((oldScan) => store.delete(oldScan.id));
    };
    request.onerror = () => transaction.abort();
    transaction.oncomplete = resolve;
    transaction.onabort = () => reject(transaction.error || new Error('Unable to save scan history.'));
    transaction.onerror = () => reject(transaction.error || new Error('Unable to save scan history.'));
  }));
}

function updateHistoryLocation(id, location) {
  return openHistoryDatabase().then((database) => new Promise((resolve, reject) => {
    const transaction = database.transaction(historyStoreName, 'readwrite');
    const store = transaction.objectStore(historyStoreName);
    const request = store.get(id);
    request.onsuccess = () => {
      if (!request.result) {
        transaction.abort();
        return;
      }
      store.put({ ...request.result, location });
    };
    request.onerror = () => transaction.abort();
    transaction.oncomplete = resolve;
    transaction.onabort = () => reject(transaction.error || new Error('Unable to save scan location.'));
    transaction.onerror = () => reject(transaction.error || new Error('Unable to save scan location.'));
  }));
}

function deleteHistoryScan(id) {
  return openHistoryDatabase().then((database) => new Promise((resolve, reject) => {
    const transaction = database.transaction(historyStoreName, 'readwrite');
    transaction.objectStore(historyStoreName).delete(id);
    transaction.oncomplete = resolve;
    transaction.onabort = () => reject(transaction.error || new Error('Unable to delete scan.'));
    transaction.onerror = () => reject(transaction.error || new Error('Unable to delete scan.'));
  }));
}

function createHistoryElement(scan) {
  const item = document.createElement('article');
  item.className = 'history-item';

  const thumbnail = document.createElement('img');
  thumbnail.className = 'history-thumb';
  thumbnail.src = scan.image;
  thumbnail.alt = `Annotated scan: ${scan.filename}`;
  item.appendChild(thumbnail);

  const info = document.createElement('div');
  info.className = 'history-info';
  const title = document.createElement('h3');
  title.textContent = scan.filename;
  info.appendChild(title);

  const metadata = document.createElement('div');
  metadata.className = 'history-meta';
  const date = document.createElement('span');
  date.textContent = new Date(scan.createdAt).toLocaleString();
  metadata.appendChild(date);

  if (scan.location) {
    const mapLink = document.createElement('a');
    mapLink.href = `https://www.google.com/maps?q=${scan.location.latitude},${scan.location.longitude}`;
    mapLink.target = '_blank';
    mapLink.rel = 'noopener noreferrer';
    mapLink.textContent = `${scan.location.latitude.toFixed(5)}, ${scan.location.longitude.toFixed(5)}`;
    metadata.appendChild(mapLink);
  } else {
    const locationLabel = document.createElement('span');
    locationLabel.textContent = 'Location not added';
    metadata.appendChild(locationLabel);
  }
  info.appendChild(metadata);

  const summary = document.createElement('p');
  summary.className = 'history-summary';
  summary.textContent = `${scan.priority} · ${scan.hazardClass} · ${scan.potholes} potholes · PSI ${scan.psi}/100 · ₹${scan.repairCost.toLocaleString('en-IN')}`;
  info.appendChild(summary);
  item.appendChild(info);

  const actions = document.createElement('div');
  actions.className = 'history-actions';
  if (!scan.location) {
    const locationButton = document.createElement('button');
    locationButton.className = 'history-location';
    locationButton.type = 'button';
    locationButton.dataset.action = 'locate';
    locationButton.dataset.id = scan.id;
    locationButton.textContent = 'Add location';
    actions.appendChild(locationButton);
  }

  const deleteButton = document.createElement('button');
  deleteButton.className = 'history-delete';
  deleteButton.type = 'button';
  deleteButton.dataset.action = 'delete';
  deleteButton.dataset.id = scan.id;
  deleteButton.textContent = 'Delete';
  actions.appendChild(deleteButton);
  item.appendChild(actions);

  return item;
}

async function renderHistory() {
  currentScans = await getHistoryScans();
  historyCount.textContent = `${currentScans.length} ${currentScans.length === 1 ? 'scan' : 'scans'}`;
  exportCsvButton.disabled = currentScans.length === 0;
  exportPdfButton.disabled = currentScans.length === 0;
  historyList.replaceChildren();
  updateScanMap(currentScans);

  if (!currentScans.length) {
    const emptyState = document.createElement('div');
    emptyState.className = 'history-empty';
    emptyState.textContent = 'Your completed inspections will appear here.';
    historyList.appendChild(emptyState);
    return;
  }
  currentScans.forEach((scan) => historyList.appendChild(createHistoryElement(scan)));
}

function csvCell(value) {
  let text = value === null || value === undefined ? '' : String(value);
  if (/^[\t\r ]*[=+\-@]/.test(text)) text = `'${text}`;
  return `"${text.replace(/"/g, '""')}"`;
}

function exportHistoryCsv() {
  if (!currentScans.length) return;

  const columns = [
    ['scanned_at', (scan) => scan.createdAt],
    ['filename', (scan) => scan.filename],
    ['hazard_class', (scan) => scan.hazardClass],
    ['priority', (scan) => scan.priority],
    ['potholes', (scan) => scan.potholes],
    ['psi', (scan) => scan.psi],
    ['max_depth_cm', (scan) => scan.maxDepthCm],
    ['action_advisory', (scan) => scan.actionAdvisory],
    ['estimated_repair_cost_inr', (scan) => scan.repairCost],
    ['gps_latitude', (scan) => isValidLocation(scan.location) ? scan.location.latitude : ''],
    ['gps_longitude', (scan) => isValidLocation(scan.location) ? scan.location.longitude : '']
  ];
  const lines = [
    columns.map(([name]) => csvCell(name)).join(','),
    ...currentScans.map((scan) => columns.map(([, getValue]) => csvCell(getValue(scan))).join(','))
  ];
  const file = new Blob([`\uFEFF${lines.join('\r\n')}`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = url;
  link.download = `roadguard-scans-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  historyMessage.textContent = `Exported ${currentScans.length} scan${currentScans.length === 1 ? '' : 's'} as CSV.`;
}

function appendReportText(documentRef, parent, tagName, text, className) {
  const element = documentRef.createElement(tagName);
  element.textContent = text;
  if (className) element.className = className;
  parent.appendChild(element);
  return element;
}

function exportHistoryPdf() {
  if (!currentScans.length) return;

  const reportWindow = window.open('', '_blank');
  if (!reportWindow) {
    historyMessage.textContent = 'Report window was blocked. Allow pop-ups to print or save a PDF report.';
    return;
  }

  const reportDocument = reportWindow.document;
  reportDocument.open();
  reportDocument.write(`<!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>RoadGuard Inspection Report</title>
        <style>
          * { box-sizing: border-box; }
          body { margin: 0; padding: 32px; color: #172033; font: 14px/1.5 Arial, sans-serif; }
          main { max-width: 900px; margin: auto; }
          .print-button { display: block; margin: 0 0 22px auto; padding: 10px 16px; border: 0; border-radius: 6px; color: #fff; background: #146f73; cursor: pointer; font: 700 13px Arial, sans-serif; }
          h1 { margin: 0 0 4px; font-size: 26px; }
          .subtitle { margin: 0 0 24px; color: #58667a; }
          .scan { display: grid; grid-template-columns: minmax(0, 1fr) minmax(180px, 42%); gap: 20px; padding: 18px 0; border-top: 1px solid #d8dee8; break-inside: avoid; }
          h2 { margin: 0 0 8px; overflow-wrap: anywhere; font-size: 17px; }
          .details { margin: 0; padding: 0; list-style: none; }
          .details li { margin: 5px 0; overflow-wrap: anywhere; }
          .scan img { display: block; width: 100%; max-height: 270px; object-fit: contain; background: #f0f3f7; }
          .footer { margin-top: 20px; color: #58667a; font-size: 11px; }
          @media print { body { padding: 0; } .print-button { display: none; } }
        </style>
      </head>
      <body><button id="printReport" class="print-button" type="button">Print / Save as PDF</button><main></main></body>
    </html>`);
  reportDocument.close();

  const main = reportDocument.querySelector('main');
  appendReportText(reportDocument, main, 'h1', 'RoadGuard Inspection Report');
  appendReportText(
    reportDocument,
    main,
    'p',
    `${currentScans.length} recent scan${currentScans.length === 1 ? '' : 's'} · Generated ${new Date().toLocaleString()}`,
    'subtitle'
  );

  currentScans.forEach((scan) => {
    const section = reportDocument.createElement('section');
    section.className = 'scan';
    const details = reportDocument.createElement('div');
    appendReportText(reportDocument, details, 'h2', scan.filename);
    const list = reportDocument.createElement('ul');
    list.className = 'details';
    const locationText = isValidLocation(scan.location)
      ? `${scan.location.latitude.toFixed(5)}, ${scan.location.longitude.toFixed(5)}`
      : 'Not added';
    [
      `Scanned: ${new Date(scan.createdAt).toLocaleString()}`,
      `Result: ${scan.hazardClass}`,
      `Priority: ${scan.priority}`,
      `Potholes: ${scan.potholes}`,
      `PSI: ${scan.psi}/100`,
      `Maximum depth: ${scan.maxDepthCm ?? '—'} cm`,
      `Estimated repair: ₹${Number(scan.repairCost || 0).toLocaleString('en-IN')}`,
      `Location: ${locationText}`,
      `Recommended action: ${scan.actionAdvisory || '—'}`
    ].forEach((value) => appendReportText(reportDocument, list, 'li', value));
    details.appendChild(list);
    section.appendChild(details);

    if (typeof scan.image === 'string' && /^data:image\/(?:jpeg|png);base64,/.test(scan.image)) {
      const image = reportDocument.createElement('img');
      image.src = scan.image;
      image.alt = `Annotated image for ${scan.filename}`;
      section.appendChild(image);
    }

    main.appendChild(section);
  });

  appendReportText(reportDocument, main, 'p', 'Generated locally by RoadGuard.', 'footer');
  historyMessage.textContent = 'Report opened in a new tab. Choose “Print / Save as PDF” there to export.';
  reportDocument.getElementById('printReport').addEventListener('click', () => reportWindow.print());
  const imageReady = Promise.all(
    Array.from(reportDocument.images, (image) => image.decode().catch(() => undefined))
  );
  imageReady.then(() => {
    if (!reportWindow.closed) reportWindow.focus();
  });
}

exportCsvButton.addEventListener('click', exportHistoryCsv);
exportPdfButton.addEventListener('click', exportHistoryPdf);

function showLocationError(error) {
  if (error.code === 1) return 'Location permission was denied. Allow location access in your browser settings to add it.';
  if (error.code === 2) return 'Your location could not be determined. Try again when GPS is available.';
  if (error.code === 3) return 'Location request timed out. Please try again.';
  return error.message || 'Unable to update scan history.';
}

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

function validateImageFile(file) {
  const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  const expectedType = allowedImageTypes.get(extension);
  if (!expectedType) return 'Unsupported file type. Choose a JPG, PNG, or WebP image.';
  if (file.type && file.type !== expectedType) return 'The file type does not match its filename. Choose a valid JPG, PNG, or WebP image.';
  if (file.size === 0) return 'This file is empty. Choose another image.';
  if (file.size > maxUploadBytes) return 'This image is larger than 10 MB. Choose a smaller image.';
  return '';
}

function setUploadMessage(message, isError = false) {
  uploadMessage.textContent = message;
  uploadMessage.classList.toggle('error', isError);
}

function setUploadProgress(message, percent, indeterminate = false) {
  uploadProgressText.textContent = message;
  uploadProgressTrack.classList.toggle('indeterminate', indeterminate);
  uploadProgressPercent.textContent = indeterminate ? '…' : `${percent}%`;
  uploadProgressTrack.setAttribute('aria-valuenow', String(percent));
  uploadProgressBar.style.width = indeterminate ? '' : `${percent}%`;
}

function requestDetection(file) {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append('file', file);

    const request = new XMLHttpRequest();
    request.open('POST', '/api/v1/detect');
    request.timeout = 180000;
    request.upload.addEventListener('progress', (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
      setUploadProgress('Uploading image…', percent);
    });
    request.upload.addEventListener('load', () => {
      setUploadProgress('Image uploaded. Analyzing road surface…', 100, true);
    });
    request.addEventListener('load', () => {
      let data;
      try {
        data = JSON.parse(request.responseText);
      } catch {
        reject(new Error('The server returned an unreadable response. Please try again.'));
        return;
      }

      if (request.status < 200 || request.status >= 300) {
        reject(new Error(data.detail || 'Image analysis failed. Please try again.'));
        return;
      }
      resolve(data);
    });
    request.addEventListener('error', () => reject(new Error('Could not reach the detection server. Check your connection and try again.')));
    request.addEventListener('timeout', () => reject(new Error('Image analysis timed out. Try again with a smaller image.')));
    request.addEventListener('abort', () => reject(new Error('Image upload was cancelled.')));
    request.send(formData);
  });
}

async function analyzeFile(file) {
  if (!file) return;

  const validationError = validateImageFile(file);
  if (validationError) {
    setUploadMessage(validationError, true);
    cameraInput.value = '';
    return;
  }

  setUploadMessage('');
  setUploadProgress('Preparing image…', 0);
  previewCard.classList.add('has-image');
  previewPlaceholder.style.display = 'none';
  loadingSpinner.classList.add('visible');
  resultCard.classList.add('loading');

  try {
    const previewData = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') resolve(reader.result);
        else reject(new Error('Unable to preview this image'));
      };
      reader.onerror = () => reject(reader.error || new Error('Unable to read this image'));
      reader.readAsDataURL(file);
    });
    imagePreview.src = previewData;

    const data = await requestDetection(file);
    if (!data.success || !data.annotated_image) {
      throw new Error('The server response did not include an annotated image. Please try again.');
    }

    imagePreview.src = data.annotated_image;
    setUploadProgress('Analysis complete.', 100);
    loadingSpinner.classList.remove('visible');
    resultCard.classList.remove('loading');

    if (data.success) {
      const sev = data.severity_assessment;
      priorityBadge.innerText = sev.priority_level;
      priorityBadge.classList.remove('priority-p1', 'priority-p2', 'priority-p3');
      if (sev.priority_level.includes('P1')) {
        priorityBadge.classList.add('priority-p1');
      } else if (sev.priority_level.includes('P2')) {
        priorityBadge.classList.add('priority-p2');
      } else {
        priorityBadge.classList.add('priority-p3');
      }

      defectTitle.innerText = `${sev.hazard_class}`;
      statPotholes.innerText = sev.num_potholes;
      statPsi.innerText = `${sev.psi}/100`;
      statDepth.innerText = `${sev.max_depth_cm} cm`;
      actionText.innerText = `Action: ${sev.action_advisory}`;
      costText.innerText = `Estimated Repair: ₹ ${sev.estimated_repair_cost_inr.toLocaleString()} INR`;

      const scan = {
        id: window.crypto && crypto.randomUUID
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        createdAt: new Date().toISOString(),
        filename: file.name || 'Road image',
        image: data.annotated_image,
        hazardClass: sev.hazard_class,
        priority: sev.priority_level,
        potholes: sev.num_potholes,
        psi: sev.psi,
        maxDepthCm: sev.max_depth_cm,
        actionAdvisory: sev.action_advisory,
        repairCost: sev.estimated_repair_cost_inr,
        location: null
      };
      try {
        await saveHistoryScan(scan);
        await renderHistory();
        historyMessage.textContent = 'Scan saved in this browser. Add location is optional.';
      } catch (error) {
        historyMessage.textContent = `Scan completed, but history could not be saved: ${error.message}`;
      }
    }
    setUploadMessage(`Analysis complete for ${file.name}.`);
  } catch (err) {
    setUploadProgress('Upload stopped.', 0);
    loadingSpinner.classList.remove('visible');
    resultCard.classList.remove('loading');
    defectTitle.innerText = 'Analysis unavailable';
    actionText.innerText = err.message || 'The image could not be analyzed.';
    costText.innerText = 'Please check the image and try again.';
    setUploadMessage(err.message || 'The image could not be analyzed.', true);
  } finally {
    cameraInput.value = '';
  }
}

historyList.addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;

  const { action, id } = button.dataset;
  button.disabled = true;

  if (action === 'locate') {
    if (!navigator.geolocation) {
      historyMessage.textContent = 'Location is not supported by this browser.';
      button.disabled = false;
      return;
    }

    button.textContent = 'Locating…';
    navigator.geolocation.getCurrentPosition(async (position) => {
      try {
        await updateHistoryLocation(id, {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude
        });
        await renderHistory();
        historyMessage.textContent = 'Location saved with this scan.';
      } catch (error) {
        historyMessage.textContent = `Location could not be saved: ${error.message}`;
        button.disabled = false;
        button.textContent = 'Add location';
      }
    }, (error) => {
      historyMessage.textContent = showLocationError(error);
      button.disabled = false;
      button.textContent = 'Add location';
    }, { enableHighAccuracy: false, timeout: 10000, maximumAge: 60000 });
    return;
  }

  if (action === 'delete') {
    try {
      await deleteHistoryScan(id);
      await renderHistory();
      historyMessage.textContent = 'Scan deleted from this browser.';
    } catch (error) {
      historyMessage.textContent = `Scan could not be deleted: ${error.message}`;
      button.disabled = false;
    }
  }
});

updateSystemStatus();
updateAuthStatus();
initializeScanMap();
renderHistory().catch((error) => {
  historyMessage.textContent = error.message;
  historyList.replaceChildren();
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js').catch(() => {});
}
