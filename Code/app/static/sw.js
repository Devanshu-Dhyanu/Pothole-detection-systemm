const CACHE_NAME = 'roadguard-v1';
const ASSETS = [
  '/static/index.html',
  '/static/styles.css',
  '/static/app.js',
  '/static/manifest.json',
  '/static/icons/icon-192.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then((res) => res || fetch(e.request))
  );
});

