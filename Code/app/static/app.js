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
const historyDatabaseName = 'roadguard-history';
const historyStoreName = 'scans';
const maxHistoryEntries = 20;
let scanMap = null;
let scanMarkerLayer = null;

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
  const scans = await getHistoryScans();
  historyCount.textContent = `${scans.length} ${scans.length === 1 ? 'scan' : 'scans'}`;
  historyList.replaceChildren();
  updateScanMap(scans);

  if (!scans.length) {
    const emptyState = document.createElement('div');
    emptyState.className = 'history-empty';
    emptyState.textContent = 'Your completed inspections will appear here.';
    historyList.appendChild(emptyState);
    return;
  }
  scans.forEach((scan) => historyList.appendChild(createHistoryElement(scan)));
}

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

async function analyzeFile(file) {
  if (!file) return;

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

    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/api/v1/detect', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error('Analysis request failed');
    const data = await response.json();
    if (!data.success || !data.annotated_image) {
      throw new Error('Analysis response did not include an annotated image');
    }

    imagePreview.src = data.annotated_image;
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
  } catch (err) {
    loadingSpinner.classList.remove('visible');
    resultCard.classList.remove('loading');
    defectTitle.innerText = 'Analysis unavailable';
    actionText.innerText = 'Detection server is offline or the image could not be processed.';
    costText.innerText = 'Please try again.';
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

initializeScanMap();
renderHistory().catch((error) => {
  historyMessage.textContent = error.message;
  historyList.replaceChildren();
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js').catch(() => {});
}
