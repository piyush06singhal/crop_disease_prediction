// PWA functionality for Crop Disease Prediction System
// Handles service worker registration, install prompt, and offline detection

class PWAHandler {
  constructor() {
    this.deferredPrompt = null;
    this.isOnline = navigator.onLine;
    this.init();
  }

  init() {
    this.registerServiceWorker();
    this.setupInstallPrompt();
    this.setupNetworkDetection();
    this.setupOfflineStorage();
  }

  // Register service worker
  registerServiceWorker() {
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/js/sw.js')
          .then(registration => {
            console.log('[PWA] Service Worker registered:', registration.scope);

            // Handle updates
            registration.addEventListener('updatefound', () => {
              const newWorker = registration.installing;
              newWorker.addEventListener('statechange', () => {
                if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                  this.showUpdateNotification();
                }
              });
            });
          })
          .catch(error => {
            console.error('[PWA] Service Worker registration failed:', error);
          });
      });
    }
  }

  // Setup install prompt
  setupInstallPrompt() {
    window.addEventListener('beforeinstallprompt', (e) => {
      console.log('[PWA] Install prompt triggered');
      e.preventDefault();
      this.deferredPrompt = e;
      this.showInstallButton();
    });

    window.addEventListener('appinstalled', () => {
      console.log('[PWA] App installed');
      this.hideInstallButton();
      this.deferredPrompt = null;
    });
  }

  // Setup network detection
  setupNetworkDetection() {
    window.addEventListener('online', () => {
      console.log('[PWA] Back online');
      this.isOnline = true;
      this.hideOfflineIndicator();
      this.syncOfflineData();
    });

    window.addEventListener('offline', () => {
      console.log('[PWA] Gone offline');
      this.isOnline = false;
      this.showOfflineIndicator();
    });
  }

  // Setup offline storage
  setupOfflineStorage() {
    // Initialize IndexedDB for offline predictions
    this.initIndexedDB();
  }

  // Initialize IndexedDB
  initIndexedDB() {
    const request = indexedDB.open('CropDiseaseDB', 1);

    request.onerror = () => {
      console.error('[PWA] IndexedDB error:', request.error);
    };

    request.onsuccess = () => {
      console.log('[PWA] IndexedDB initialized');
      this.db = request.result;
    };

    request.onupgradeneeded = (event) => {
      const db = event.target.result;

      // Create predictions store
      if (!db.objectStoreNames.contains('predictions')) {
        const predictionsStore = db.createObjectStore('predictions', { keyPath: 'id' });
        predictionsStore.createIndex('timestamp', 'timestamp', { unique: false });
      }

      // Create images store for offline storage
      if (!db.objectStoreNames.contains('images')) {
        db.createObjectStore('images', { keyPath: 'id' });
      }
    };
  }

  // Show install button
  showInstallButton() {
    const installButton = document.createElement('button');
    installButton.id = 'pwa-install-btn';
    installButton.className = 'fixed bottom-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg hover:bg-blue-700 transition-colors z-50';
    installButton.innerHTML = `
      <svg class="w-5 h-5 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
      </svg>
      Install App
    `;

    installButton.addEventListener('click', () => {
      this.installApp();
    });

    document.body.appendChild(installButton);
  }

  // Hide install button
  hideInstallButton() {
    const installButton = document.getElementById('pwa-install-btn');
    if (installButton) {
      installButton.remove();
    }
  }

  // Install the app
  async installApp() {
    if (!this.deferredPrompt) return;

    this.deferredPrompt.prompt();
    const { outcome } = await this.deferredPrompt.userChoice;

    console.log('[PWA] Install outcome:', outcome);
    this.deferredPrompt = null;
    this.hideInstallButton();
  }

  // Show offline indicator
  showOfflineIndicator() {
    let indicator = document.getElementById('offline-indicator');
    if (!indicator) {
      indicator = document.createElement('div');
      indicator.id = 'offline-indicator';
      indicator.className = 'fixed top-0 left-0 right-0 bg-yellow-500 text-white text-center py-2 z-50';
      indicator.innerHTML = `
        <div class="flex items-center justify-center">
          <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
          </svg>
          You're offline. Some features may be limited.
        </div>
      `;
      document.body.appendChild(indicator);
    }
  }

  // Hide offline indicator
  hideOfflineIndicator() {
    const indicator = document.getElementById('offline-indicator');
    if (indicator) {
      indicator.remove();
    }
  }

  // Show update notification
  showUpdateNotification() {
    const notification = document.createElement('div');
    notification.id = 'update-notification';
    notification.className = 'fixed top-4 right-4 bg-blue-600 text-white px-4 py-3 rounded-lg shadow-lg z-50 max-w-sm';
    notification.innerHTML = `
      <div class="flex items-start">
        <div class="flex-1">
          <p class="font-medium">Update Available</p>
          <p class="text-sm opacity-90">A new version is available. Refresh to update.</p>
        </div>
        <button id="update-btn" class="ml-3 bg-blue-700 hover:bg-blue-800 px-3 py-1 rounded text-sm transition-colors">
          Update
        </button>
        <button id="dismiss-update" class="ml-2 text-blue-200 hover:text-white">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      </div>
    `;

    document.body.appendChild(notification);

    // Handle update button
    document.getElementById('update-btn').addEventListener('click', () => {
      window.location.reload();
    });

    // Handle dismiss button
    document.getElementById('dismiss-update').addEventListener('click', () => {
      notification.remove();
    });
  }

  // Store prediction offline
  async storePredictionOffline(predictionData) {
    if (!this.db) return false;

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['predictions'], 'readwrite');
      const store = transaction.objectStore('predictions');

      const prediction = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        ...predictionData,
        synced: false
      };

      const request = store.add(prediction);
      request.onsuccess = () => {
        console.log('[PWA] Prediction stored offline:', prediction.id);
        resolve(true);
      };
      request.onerror = () => reject(request.error);
    });
  }

  // Sync offline data when back online
  async syncOfflineData() {
    if (!this.db || !this.isOnline) return;

    try {
      const transaction = this.db.transaction(['predictions'], 'readonly');
      const store = transaction.objectStore('predictions');
      const index = store.index('timestamp');

      const request = index.getAll();
      request.onsuccess = async () => {
        const predictions = request.result.filter(p => !p.synced);

        for (const prediction of predictions) {
          try {
            // Send to server
            const response = await fetch('/api/v1/predict/sync', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(prediction)
            });

            if (response.ok) {
              // Mark as synced
              await this.markPredictionSynced(prediction.id);
            }
          } catch (error) {
            console.error('[PWA] Sync failed for prediction:', prediction.id, error);
          }
        }
      };
    } catch (error) {
      console.error('[PWA] Sync error:', error);
    }
  }

  // Mark prediction as synced
  async markPredictionSynced(id) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['predictions'], 'readwrite');
      const store = transaction.objectStore('predictions');

      const getRequest = store.get(id);
      getRequest.onsuccess = () => {
        const prediction = getRequest.result;
        prediction.synced = true;

        const updateRequest = store.put(prediction);
        updateRequest.onsuccess = () => resolve();
        updateRequest.onerror = () => reject(updateRequest.error);
      };
      getRequest.onerror = () => reject(getRequest.error);
    });
  }

  // Check if running as PWA
  isPWA() {
    return window.matchMedia('(display-mode: standalone)').matches ||
           window.navigator.standalone === true;
  }

  // Get network status
  getNetworkStatus() {
    return {
      online: this.isOnline,
      connection: navigator.connection || navigator.mozConnection || navigator.webkitConnection
    };
  }
}

// Initialize PWA handler when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.pwaHandler = new PWAHandler();
});