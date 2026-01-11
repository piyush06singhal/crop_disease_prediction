// frontend/static/js/app.js - Enhanced frontend application logic
/**
 * Crop Disease Prediction System - Frontend JavaScript
 * Advanced features: Camera access, PWA, offline support, progress tracking
 */

class CropDiseasePredictor {
    constructor() {
        this.sessionId = null;
        this.currentQuestions = [];
        this.selectedFile = null;
        this.cameraStream = null;
        this.facingMode = 'environment'; // Back camera by default
        this.isOnline = navigator.onLine;
        this.deferredPrompt = null;
        this.cacheName = 'crop-disease-v1';

        this.init();
    }

    init() {
        this.bindEvents();
        this.setupPWA();
        this.setupOfflineDetection();
        this.checkHealth();
        this.loadHistory();
        this.setupCamera();
    }

    bindEvents() {
        // Tab switching
        document.getElementById('upload-tab').addEventListener('click', () => this.switchTab('upload'));
        document.getElementById('camera-tab').addEventListener('click', () => this.switchTab('camera'));

        // Upload events
        const uploadArea = document.getElementById('upload-area');
        const imageInput = document.getElementById('image-input');

        uploadArea.addEventListener('click', () => imageInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        imageInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Camera events
        document.getElementById('camera-toggle').addEventListener('click', this.toggleCamera.bind(this));
        document.getElementById('capture-btn').addEventListener('click', this.captureImage.bind(this));
        document.getElementById('switch-camera').addEventListener('click', this.switchCamera.bind(this));

        // Analysis and interaction events
        document.getElementById('analyze-btn').addEventListener('click', this.analyzeImage.bind(this));

        // PWA events
        document.getElementById('install-dismiss').addEventListener('click', () => this.hideInstallPrompt());
        document.getElementById('install-accept').addEventListener('click', () => this.installPWA());

        // History events
        document.getElementById('refresh-history').addEventListener('click', () => this.loadHistory());

        // Share and download events
        document.getElementById('share-results').addEventListener('click', () => this.shareResults());
        document.getElementById('download-report').addEventListener('click', () => this.downloadReport());

        // Online/offline events
        window.addEventListener('online', () => this.handleOnline());
        window.addEventListener('offline', () => this.handleOffline());

        // PWA install prompt
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            this.deferredPrompt = e;
            this.showInstallPrompt();
        });

        // App installed
        window.addEventListener('appinstalled', () => {
            this.hideInstallPrompt();
            this.showToast('App installed successfully!', 'success');
        });
    }

    setupPWA() {
        // Register for periodic background sync if supported
        if ('serviceWorker' in navigator && 'periodicSync' in window.ServiceWorkerRegistration.prototype) {
            navigator.serviceWorker.ready.then(registration => {
                // Request permission for background sync
                return navigator.permissions.query({ name: 'periodic-background-sync' });
            }).then(permission => {
                if (permission.state === 'granted') {
                    // Register for periodic sync every 24 hours
                    return navigator.serviceWorker.ready;
                }
            }).then(registration => {
                if (registration) {
                    registration.periodicSync.register('update-models', {
                        minInterval: 24 * 60 * 60 * 1000 // 24 hours
                    });
                }
            }).catch(err => console.log('Periodic sync not available'));
        }
    }

    setupOfflineDetection() {
        this.updateOnlineStatus();
    }

    switchTab(tab) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active', 'text-blue-600', 'border-b-2', 'border-blue-600');
            btn.classList.add('text-gray-500');
        });

        document.getElementById(`${tab}-tab`).classList.add('active', 'text-blue-600', 'border-b-2', 'border-blue-600');

        // Show/hide sections
        document.getElementById('upload-section').classList.toggle('hidden', tab !== 'upload');
        document.getElementById('camera-section').classList.toggle('hidden', tab !== 'camera');

        // Stop camera if switching away
        if (tab !== 'camera' && this.cameraStream) {
            this.stopCamera();
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    async handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showToast('Please select a valid image file', 'error');
            return;
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            this.showToast('File size must be less than 16MB', 'error');
            return;
        }

        this.selectedFile = file;

        // Show upload progress
        this.showUploadProgress();

        // Simulate upload progress
        await this.simulateUploadProgress();

        // Show preview
        await this.showImagePreview(file);
    }

    showUploadProgress() {
        document.getElementById('upload-content').classList.add('hidden');
        document.getElementById('upload-progress').classList.remove('hidden');
        document.getElementById('upload-area').classList.add('uploading');
    }

    hideUploadProgress() {
        document.getElementById('upload-content').classList.remove('hidden');
        document.getElementById('upload-progress').classList.add('hidden');
        document.getElementById('upload-area').classList.remove('uploading');
    }

    async simulateUploadProgress() {
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');

        for (let i = 0; i <= 100; i += 10) {
            progressBar.style.width = `${i}%`;
            progressText.textContent = `Uploading... ${i}%`;
            await this.delay(100);
        }
    }

    async showImagePreview(file) {
        this.hideUploadProgress();

        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('image-preview');
            preview.src = e.target.result;

            // Analyze image metadata
            this.analyzeImageMetadata(file);

            document.getElementById('preview-section').classList.remove('hidden');
            document.getElementById('analyze-btn').disabled = false;
        };
        reader.readAsDataURL(file);
    }

    analyzeImageMetadata(file) {
        // Get basic file info
        document.getElementById('image-size').textContent = `Size: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        document.getElementById('image-format').textContent = `Format: ${file.type.split('/')[1].toUpperCase()}`;

        // Get image dimensions
        const img = new Image();
        img.onload = () => {
            document.getElementById('image-dimensions').textContent = `Dimensions: ${img.width} × ${img.height}`;
        };
        img.src = URL.createObjectURL(file);
    }

    setupCamera() {
        // Check camera support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            document.getElementById('camera-toggle').disabled = true;
            document.getElementById('camera-toggle').textContent = 'Camera not supported';
            return;
        }
    }

    async toggleCamera() {
        if (this.cameraStream) {
            this.stopCamera();
        } else {
            await this.startCamera();
        }
    }

    async startCamera() {
        try {
            const constraints = {
                video: {
                    facingMode: this.facingMode,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                },
                audio: false
            };

            this.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            const videoElement = document.getElementById('camera-feed');
            videoElement.srcObject = this.cameraStream;

            document.getElementById('camera-toggle').textContent = 'Stop Camera';
            document.getElementById('switch-camera').classList.remove('hidden');

            // Check for multiple cameras
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            if (videoDevices.length < 2) {
                document.getElementById('switch-camera').classList.add('hidden');
            }

        } catch (error) {
            console.error('Camera access error:', error);
            this.showToast('Unable to access camera. Please check permissions.', 'error');
        }
    }

    stopCamera() {
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(track => track.stop());
            this.cameraStream = null;
        }

        const videoElement = document.getElementById('camera-feed');
        videoElement.srcObject = null;

        document.getElementById('camera-toggle').textContent = 'Start Camera';
        document.getElementById('switch-camera').classList.add('hidden');
    }

    async switchCamera() {
        this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';
        if (this.cameraStream) {
            this.stopCamera();
            await this.startCamera();
        }
    }

    async captureImage() {
        if (!this.cameraStream) return;

        const video = document.getElementById('camera-feed');
        const canvas = document.getElementById('capture-canvas');
        const context = canvas.getContext('2d');

        // Set canvas size to video size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame to canvas
        context.drawImage(video, 0, 0);

        // Add shutter effect
        this.showShutterEffect();

        // Convert to blob
        canvas.toBlob(async (blob) => {
            const file = new File([blob], `capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
            this.selectedFile = file;

            // Switch to upload tab and show preview
            this.switchTab('upload');
            await this.showImagePreview(file);

            this.showToast('Image captured successfully!', 'success');
        }, 'image/jpeg', 0.9);
    }

    showShutterEffect() {
        const shutter = document.createElement('div');
        shutter.className = 'shutter';
        document.querySelector('.camera-container').appendChild(shutter);

        setTimeout(() => {
            shutter.remove();
        }, 300);
    }

    async analyzeImage() {
        if (!this.selectedFile) {
            this.showToast('Please select an image first', 'error');
            return;
        }

        if (!this.isOnline) {
            this.showToast('You are offline. Please check your connection.', 'error');
            return;
        }

        // Disable button and show progress
        const analyzeBtn = document.getElementById('analyze-btn');
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<div class="spinner mr-2"></div> Analyzing...';

        // Show analysis progress
        this.showAnalysisProgress();

        try {
            // Simulate progress updates
            await this.simulateAnalysisProgress();

            const formData = new FormData();
            formData.append('image', this.selectedFile);

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.sessionId = result.data.session_id;
                this.displayResults(result.data);

                if (result.data.questions && result.data.questions.length > 0) {
                    this.displayQuestions(result.data.questions);
                }

                // Cache result for offline access
                await this.cacheResult(result.data);

                this.showToast('Analysis completed successfully!', 'success');

            } else {
                throw new Error(result.error?.message || 'Analysis failed');
            }

        } catch (error) {
            console.error('Analysis error:', error);
            this.showToast('Failed to analyze image. Please try again.', 'error');
        } finally {
            // Re-enable button
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `
                <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                </svg>
                Analyze Disease
            `;

            // Hide progress
            this.hideAnalysisProgress();
        }
    }

    showAnalysisProgress() {
        document.getElementById('analysis-progress').classList.remove('hidden');
    }

    hideAnalysisProgress() {
        document.getElementById('analysis-progress').classList.add('hidden');
    }

    async simulateAnalysisProgress() {
        const steps = [
            { step: 'Preprocessing image', description: 'Analyzing image quality and preparing for AI processing...', duration: 1000 },
            { step: 'AI Analysis', description: 'Running deep learning models for disease detection...', duration: 2000 },
            { step: 'Confidence Calculation', description: 'Computing prediction confidence and uncertainty...', duration: 1000 },
            { step: 'Generating Report', description: 'Creating detailed analysis report...', duration: 1000 }
        ];

        let totalProgress = 0;
        const progressBar = document.getElementById('analysis-progress-bar');
        const percentageEl = document.getElementById('progress-percentage');
        const stepEl = document.getElementById('current-step');
        const descriptionEl = document.getElementById('progress-description');

        for (const step of steps) {
            stepEl.textContent = step.step;
            descriptionEl.textContent = step.description;

            const stepProgress = 100 / steps.length;
            for (let i = 0; i <= stepProgress; i += 2) {
                totalProgress = Math.min(totalProgress + 2, 100);
                progressBar.style.width = `${totalProgress}%`;
                percentageEl.textContent = `${Math.round(totalProgress)}%`;
                await this.delay(50);
            }

            await this.delay(step.duration - 1000); // Account for progress animation
        }
    }

    displayResults(data) {
        const resultsContent = document.getElementById('results-content');

        // Create comprehensive results display
        const html = `
            <div class="grid md:grid-cols-2 gap-6">
                <!-- Main Prediction -->
                <div class="space-y-4">
                    <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                        <div class="flex items-center mb-2">
                            <svg class="w-5 h-5 text-green-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            <h3 class="text-lg font-semibold text-green-800">Primary Diagnosis</h3>
                        </div>
                        <p class="text-2xl font-bold text-green-900 mb-1">${data.predictions[0]?.disease || 'Unknown'}</p>
                        <p class="text-green-700">Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
                        <p class="text-sm text-green-600 mt-2">Crop: ${data.crop_type || 'Detected automatically'}</p>
                    </div>

                    <!-- Confidence Breakdown -->
                    <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 class="font-semibold text-blue-800 mb-3">Confidence Analysis</h4>
                        <div class="space-y-2">
                            <div class="flex justify-between items-center">
                                <span class="text-sm text-blue-700">Image Analysis</span>
                                <div class="flex items-center">
                                    <div class="w-24 bg-blue-200 rounded-full h-2 mr-2">
                                        <div class="bg-blue-600 h-2 rounded-full" style="width: ${(data.confidence_breakdown?.image_prediction * 100) || 0}%"></div>
                                    </div>
                                    <span class="text-sm font-medium">${((data.confidence_breakdown?.image_prediction * 100) || 0).toFixed(1)}%</span>
                                </div>
                            </div>
                            <div class="flex justify-between items-center">
                                <span class="text-sm text-blue-700">Crop Validation</span>
                                <div class="flex items-center">
                                    <div class="w-24 bg-blue-200 rounded-full h-2 mr-2">
                                        <div class="bg-blue-600 h-2 rounded-full" style="width: ${(data.confidence_breakdown?.crop_validation * 100) || 0}%"></div>
                                    </div>
                                    <span class="text-sm font-medium">${((data.confidence_breakdown?.crop_validation * 100) || 0).toFixed(1)}%</span>
                                </div>
                            </div>
                            <div class="flex justify-between items-center">
                                <span class="text-sm text-blue-700">Q&A Reasoning</span>
                                <div class="flex items-center">
                                    <div class="w-24 bg-blue-200 rounded-full h-2 mr-2">
                                        <div class="bg-blue-600 h-2 rounded-full" style="width: ${(data.confidence_breakdown?.qa_reasoning * 100) || 0}%"></div>
                                    </div>
                                    <span class="text-sm font-medium">${((data.confidence_breakdown?.qa_reasoning * 100) || 0).toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Alternative Predictions -->
                <div class="space-y-4">
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h4 class="font-semibold text-gray-800 mb-3">Alternative Possibilities</h4>
                        <div class="space-y-2">
                            ${data.predictions.slice(1, 4).map(pred => `
                                <div class="flex justify-between items-center p-2 bg-white rounded border">
                                    <span class="text-sm text-gray-700">${pred.disease}</span>
                                    <span class="text-sm font-medium text-gray-900">${(pred.confidence * 100).toFixed(1)}%</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>

                    <!-- Recommendations -->
                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                        <div class="flex items-center mb-2">
                            <svg class="w-5 h-5 text-yellow-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
                            </svg>
                            <h4 class="font-semibold text-yellow-800">Recommendations</h4>
                        </div>
                        <ul class="text-sm text-yellow-700 space-y-1">
                            <li>• Monitor affected plants closely</li>
                            <li>• Isolate infected plants to prevent spread</li>
                            <li>• Consider appropriate treatment options</li>
                            <li>• Consult local agricultural extension service</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;

        resultsContent.innerHTML = html;
        document.getElementById('results-section').classList.remove('hidden');

        // Store result data for sharing/downloading
        this.currentResult = data;
    }

    displayQuestions(questions) {
        this.currentQuestions = questions;
        const questionsContent = document.getElementById('questions-content');

        const html = `
            <form id="questions-form" class="space-y-6">
                ${questions.map((question, index) => `
                    <div class="bg-gray-50 rounded-lg p-4">
                        <label class="block text-gray-800 font-medium mb-3">
                            ${index + 1}. ${question.question}
                        </label>
                        <div class="space-y-2">
                            ${question.options ? question.options.map(option => `
                                <label class="flex items-center">
                                    <input type="radio" name="answer_${index}" value="${option}" class="mr-3 text-blue-600">
                                    <span class="text-gray-700">${option}</span>
                                </label>
                            `).join('') : `
                                <textarea
                                    name="answer_${index}"
                                    class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                    rows="3"
                                    placeholder="Describe what you observe..."
                                    required
                                ></textarea>
                            `}
                        </div>
                        <input type="hidden" name="question_id_${index}" value="${question.id}">
                    </div>
                `).join('')}

                <div class="flex justify-end space-x-4">
                    <button type="button" id="skip-questions" class="px-6 py-2 text-gray-600 hover:text-gray-800">
                        Skip for now
                    </button>
                    <button type="submit" class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors inline-flex items-center">
                        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                        </svg>
                        Submit Answers
                    </button>
                </div>
            </form>
        `;

        questionsContent.innerHTML = html;
        document.getElementById('questions-section').classList.remove('hidden');

        // Bind form submission
        document.getElementById('questions-form').addEventListener('submit', this.submitAnswers.bind(this));
        document.getElementById('skip-questions').addEventListener('click', () => {
            document.getElementById('questions-section').classList.add('hidden');
            this.showToast('Questions skipped. Analysis may be less accurate.', 'warning');
        });
    }

    async submitAnswers(e) {
        e.preventDefault();

        const formData = new FormData(e.target);
        const answers = [];

        // Process form data
        for (let i = 0; i < this.currentQuestions.length; i++) {
            const answer = formData.get(`answer_${i}`);
            const questionId = formData.get(`question_id_${i}`);

            if (answer && questionId) {
                answers.push({
                    question_id: questionId,
                    answer: answer
                });
            }
        }

        if (answers.length === 0) {
            this.showToast('Please answer at least one question', 'error');
            return;
        }

        // Show loading state
        const submitBtn = e.target.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<div class="spinner mr-2"></div> Submitting...';

        try {
            // Submit first answer (simplified - in production, handle multiple)
            const response = await fetch('/api/answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    question_id: answers[0].question_id,
                    answer: answers[0].answer
                })
            });

            const result = await response.json();

            if (result.success) {
                this.displayResults(result.data);
                // Hide questions if confidence is high enough
                if (result.data.refined_confidence > 0.9) {
                    document.getElementById('questions-section').classList.add('hidden');
                }
                this.showToast('Answers submitted successfully!', 'success');
            } else {
                throw new Error(result.error?.message || 'Submission failed');
            }

        } catch (error) {
            console.error('Submit error:', error);
            this.showToast('Failed to submit answers. Please try again.', 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    }

    async loadHistory() {
        try {
            const response = await fetch('/api/history?limit=10');
            const result = await response.json();

            if (result.success) {
                this.displayHistory(result.data.history);
            }
        } catch (error) {
            console.warn('Failed to load history:', error);
            // Try to load from cache
            await this.loadHistoryFromCache();
        }
    }

    displayHistory(history) {
        const historyContent = document.getElementById('history-content');

        if (!history || history.length === 0) {
            historyContent.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <svg class="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <p>No analysis history yet</p>
                    <p class="text-sm">Your previous analyses will appear here</p>
                </div>
            `;
            return;
        }

        const html = history.map(item => `
            <div class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div class="flex items-start justify-between">
                    <div class="flex-1">
                        <div class="flex items-center space-x-2 mb-2">
                            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                ${item.predictions?.[0]?.disease || 'Unknown'}
                            </span>
                            <span class="text-sm text-gray-500">
                                ${(item.confidence * 100).toFixed(1)}% confidence
                            </span>
                        </div>
                        <p class="text-sm text-gray-600">
                            ${new Date(item.timestamp).toLocaleDateString()} at ${new Date(item.timestamp).toLocaleTimeString()}
                        </p>
                    </div>
                    <button class="text-gray-400 hover:text-gray-600 p-1" onclick="app.viewHistoryItem('${item.session_id}')">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
                        </svg>
                    </button>
                </div>
            </div>
        `).join('');

        historyContent.innerHTML = html;
        document.getElementById('history-section').classList.remove('hidden');
    }

    async loadHistoryFromCache() {
        try {
            const cache = await caches.open(this.cacheName);
            const response = await cache.match('/api/history');
            if (response) {
                const data = await response.json();
                this.displayHistory(data.history);
            }
        } catch (error) {
            console.warn('Failed to load history from cache:', error);
        }
    }

    async cacheResult(result) {
        try {
            const cache = await caches.open(this.cacheName);
            const response = new Response(JSON.stringify(result));
            await cache.put(`/result/${result.session_id}`, response);
        } catch (error) {
            console.warn('Failed to cache result:', error);
        }
    }

    shareResults() {
        if (!this.currentResult) return;

        const shareData = {
            title: 'Crop Disease Analysis',
            text: `Disease detected: ${this.currentResult.predictions[0]?.disease} (${(this.currentResult.confidence * 100).toFixed(1)}% confidence)`,
            url: window.location.href
        };

        if (navigator.share) {
            navigator.share(shareData);
        } else {
            // Fallback: copy to clipboard
            navigator.clipboard.writeText(`${shareData.text} - ${shareData.url}`);
            this.showToast('Result link copied to clipboard!', 'success');
        }
    }

    downloadReport() {
        if (!this.currentResult) return;

        const reportData = {
            timestamp: new Date().toISOString(),
            analysis: this.currentResult,
            userAgent: navigator.userAgent
        };

        const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `crop-analysis-${this.currentResult.session_id}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showToast('Report downloaded successfully!', 'success');
    }

    showInstallPrompt() {
        // Only show if not already installed and not dismissed recently
        const dismissed = localStorage.getItem('pwa-install-dismissed');
        if (dismissed && Date.now() - parseInt(dismissed) < 24 * 60 * 60 * 1000) return;

        document.getElementById('install-prompt').classList.remove('hidden');
    }

    hideInstallPrompt() {
        document.getElementById('install-prompt').classList.add('hidden');
        localStorage.setItem('pwa-install-dismissed', Date.now().toString());
    }

    async installPWA() {
        if (!this.deferredPrompt) return;

        this.deferredPrompt.prompt();
        const { outcome } = await this.deferredPrompt.userChoice;
        this.deferredPrompt = null;

        if (outcome === 'accepted') {
            this.hideInstallPrompt();
        }
    }

    handleOnline() {
        this.isOnline = true;
        document.getElementById('offline-indicator').classList.remove('show');
        this.showToast('Back online!', 'success');
        // Sync any cached data
        this.syncOfflineData();
    }

    handleOffline() {
        this.isOnline = false;
        document.getElementById('offline-indicator').classList.add('show');
        this.showToast('You are now offline', 'warning');
    }

    updateOnlineStatus() {
        if (this.isOnline) {
            document.getElementById('offline-indicator').classList.remove('show');
        } else {
            document.getElementById('offline-indicator').classList.add('show');
        }
    }

    async syncOfflineData() {
        // Sync any cached analysis requests
        try {
            const cache = await caches.open(this.cacheName);
            const keys = await cache.keys();

            for (const request of keys) {
                if (request.url.includes('/api/')) {
                    // Resubmit cached API calls
                    const response = await cache.match(request);
                    if (response) {
                        // Implement sync logic here
                    }
                }
            }
        } catch (error) {
            console.warn('Failed to sync offline data:', error);
        }
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container');

        const toast = document.createElement('div');
        toast.className = `toast max-w-sm w-full bg-white shadow-lg rounded-lg pointer-events-auto ring-1 ring-black ring-opacity-5 p-4 ${type === 'error' ? 'border-l-4 border-red-500' : type === 'success' ? 'border-l-4 border-green-500' : type === 'warning' ? 'border-l-4 border-yellow-500' : 'border-l-4 border-blue-500'}`;

        const colors = {
            error: 'text-red-600',
            success: 'text-green-600',
            warning: 'text-yellow-600',
            info: 'text-blue-600'
        };

        toast.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0">
                    <svg class="h-6 w-6 ${colors[type]}" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        ${type === 'error' ? '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />' :
                           type === 'success' ? '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />' :
                           type === 'warning' ? '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />' :
                           '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />'}
                    </svg>
                </div>
                <div class="ml-3 w-0 flex-1 pt-0.5">
                    <p class="text-sm font-medium text-gray-900">${message}</p>
                </div>
                <div class="ml-4 flex-shrink-0 flex">
                    <button class="bg-white rounded-md inline-flex text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500" onclick="this.parentElement.parentElement.parentElement.remove()">
                        <span class="sr-only">Close</span>
                        <svg class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                        </svg>
                    </button>
                </div>
            </div>
        `;

        toastContainer.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const result = await response.json();
            console.log('System health:', result);

            if (result.status !== 'healthy') {
                this.showToast('System is experiencing issues. Some features may not work.', 'warning');
            }
        } catch (error) {
            console.warn('Health check failed:', error);
            this.showToast('Unable to connect to server. Please check your connection.', 'error');
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Public method for viewing history items
    viewHistoryItem(sessionId) {
        // Navigate to result view (implement as needed)
        console.log('View history item:', sessionId);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new CropDiseasePredictor();
});