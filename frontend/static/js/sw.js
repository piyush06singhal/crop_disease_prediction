// Service Worker for Crop Disease Prediction System PWA
// Handles caching, offline functionality, and background sync

const CACHE_NAME = 'crop-disease-v1.0.0';
const STATIC_CACHE = 'crop-disease-static-v1.0.0';
const DYNAMIC_CACHE = 'crop-disease-dynamic-v1.0.0';
const TFLITE_CACHE = 'crop-disease-tflite-v1.0.0';

// Resources to cache immediately
const STATIC_ASSETS = [
  '/',
  '/static/manifest.json',
  '/static/css/styles.css',
  '/static/js/app.js',
  '/static/js/pwa.js',
  '/static/icons/icon-192x192.png',
  '/static/icons/icon-512x512.png',
  'https://cdn.tailwindcss.com',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
];

// API endpoints that can be cached
const API_CACHE_PATTERNS = [
  /\/api\/v1\/health$/,
  /\/api\/v1\/analytics\/summary$/
];

// Install event - cache static assets
self.addEventListener('install', event => {
  console.log('[SW] Installing service worker');
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('[SW] Activating service worker');
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== STATIC_CACHE &&
              cacheName !== DYNAMIC_CACHE &&
              cacheName !== TFLITE_CACHE) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Handle TensorFlow Lite model caching
  if (request.url.includes('.tflite') || request.url.includes('model')) {
    event.respondWith(
      caches.match(request)
        .then(response => {
          if (response) {
            return response;
          }

          return fetch(request).then(networkResponse => {
            if (networkResponse.ok) {
              const responseClone = networkResponse.clone();
              caches.open(TFLITE_CACHE).then(cache => {
                cache.put(request, responseClone);
              });
            }
            return networkResponse;
          });
        })
        .catch(() => {
          // Return offline fallback for models
          return new Response('Model not available offline', {
            status: 503,
            statusText: 'Service Unavailable'
          });
        })
    );
    return;
  }

  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    // Cache GET requests that match our patterns
    if (request.method === 'GET' && API_CACHE_PATTERNS.some(pattern => pattern.test(url.pathname))) {
      event.respondWith(
        caches.match(request)
          .then(response => {
            if (response) {
              return response;
            }

            return fetch(request).then(networkResponse => {
              if (networkResponse.ok) {
                const responseClone = networkResponse.clone();
                caches.open(DYNAMIC_CACHE).then(cache => {
                  cache.put(request, responseClone);
                });
              }
              return networkResponse;
            });
          })
      );
      return;
    }

    // For other API requests, try network first, then cache
    event.respondWith(
      fetch(request)
        .then(response => {
          // Cache successful responses
          if (response.ok && request.method === 'GET') {
            const responseClone = response.clone();
            caches.open(DYNAMIC_CACHE).then(cache => {
              cache.put(request, responseClone);
            });
          }
          return response;
        })
        .catch(() => {
          // Try cache for GET requests
          if (request.method === 'GET') {
            return caches.match(request);
          }
        })
    );
    return;
  }

  // Handle static assets and pages
  event.respondWith(
    caches.match(request)
      .then(response => {
        if (response) {
          return response;
        }

        return fetch(request).then(networkResponse => {
          // Cache successful GET requests
          if (networkResponse.ok && request.method === 'GET' &&
              (request.destination === 'document' ||
               request.destination === 'style' ||
               request.destination === 'script' ||
               request.destination === 'image' ||
               request.destination === 'font')) {
            const responseClone = networkResponse.clone();
            caches.open(DYNAMIC_CACHE).then(cache => {
              cache.put(request, responseClone);
            });
          }
          return networkResponse;
        });
      })
      .catch(() => {
        // Return offline fallback page
        if (request.destination === 'document') {
          return caches.match('/offline.html');
        }
      })
  );
});

// Background sync for offline predictions
self.addEventListener('sync', event => {
  console.log('[SW] Background sync triggered:', event.tag);

  if (event.tag === 'prediction-sync') {
    event.waitUntil(syncPredictions());
  }
});

// Push notifications (for future use)
self.addEventListener('push', event => {
  console.log('[SW] Push received:', event);

  if (event.data) {
    const data = event.data.json();
    const options = {
      body: data.body,
      icon: '/static/icons/icon-192x192.png',
      badge: '/static/icons/icon-72x72.png',
      vibrate: [100, 50, 100],
      data: data.data
    };

    event.waitUntil(
      self.registration.showNotification(data.title, options)
    );
  }
});

// Notification click handler
self.addEventListener('notificationclick', event => {
  console.log('[SW] Notification clicked:', event);
  event.notification.close();

  event.waitUntil(
    clients.openWindow(event.notification.data.url || '/')
  );
});

// Sync predictions function
async function syncPredictions() {
  try {
    // Get stored offline predictions
    const predictions = await getStoredPredictions();

    for (const prediction of predictions) {
      try {
        // Send to server
        const response = await fetch('/api/v1/predict/sync', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(prediction)
        });

        if (response.ok) {
          // Remove from storage
          await removeStoredPrediction(prediction.id);
        }
      } catch (error) {
        console.error('[SW] Failed to sync prediction:', prediction.id, error);
      }
    }
  } catch (error) {
    console.error('[SW] Background sync failed:', error);
  }
}

// IndexedDB helpers for offline storage
function getStoredPredictions() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('CropDiseaseDB', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['predictions'], 'readonly');
      const store = transaction.objectStore('predictions');
      const getAllRequest = store.getAll();

      getAllRequest.onerror = () => reject(getAllRequest.error);
      getAllRequest.onsuccess = () => resolve(getAllRequest.result);
    };
  });
}

function removeStoredPrediction(id) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('CropDiseaseDB', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['predictions'], 'readwrite');
      const store = transaction.objectStore('predictions');
      const deleteRequest = store.delete(id);

      deleteRequest.onerror = () => reject(deleteRequest.error);
      deleteRequest.onsuccess = () => resolve();
    };
  });
}