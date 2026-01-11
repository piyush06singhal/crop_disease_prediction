# services/background_tasks.py - Background task processing for Crop Disease Prediction System
"""
Background Task Manager - Asynchronous processing for heavy operations.

Handles background processing for:
- Batch predictions
- Model training updates
- Data analytics processing
- File cleanup operations
- Email notifications
- Report generation

Uses threading for lightweight background tasks and can be extended to Celery for distributed processing.
"""

import threading
import time
import queue
import logging
from typing import Dict, Any, Callable, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    """
    Manages background tasks with threading and queue-based processing.

    Features:
    - Task queuing and prioritization
    - Status tracking and progress monitoring
    - Error handling and retry logic
    - Graceful shutdown
    - Task result storage
    """

    def __init__(self, app=None, max_workers: int = 4):
        """
        Initialize background task manager.

        Args:
            app: Flask application instance
            max_workers: Maximum number of worker threads
        """
        self.app = app
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.workers: list[threading.Thread] = []
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()

        # Task statistics
        self.stats = {
            'total_queued': 0,
            'total_completed': 0,
            'total_failed': 0,
            'active_workers': 0
        }

    def start(self):
        """Start the background task manager"""
        logger.info(f"Starting background task manager with {self.max_workers} workers")

        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BackgroundWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info("Background task manager started successfully")

    def stop(self, timeout: int = 30):
        """Stop the background task manager gracefully"""
        logger.info("Stopping background task manager...")

        # Signal shutdown
        self.shutdown_event.set()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)

        # Cancel remaining tasks
        cancelled_count = 0
        while not self.task_queue.empty():
            try:
                _, task_id, _, _ = self.task_queue.get_nowait()
                with self.lock:
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id]['status'] = 'cancelled'
                        self.stats['total_failed'] += 1
                        cancelled_count += 1
            except queue.Empty:
                break

        logger.info(f"Background task manager stopped. Cancelled {cancelled_count} tasks.")

    def submit_task(self, task_func: Callable, *args, priority: int = 1,
                   task_type: str = 'generic', **kwargs) -> str:
        """
        Submit a task for background processing.

        Args:
            task_func: Function to execute
            *args: Positional arguments for the function
            priority: Task priority (lower number = higher priority)
            task_type: Type of task for categorization
            **kwargs: Keyword arguments for the function

        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())

        task_data = {
            'id': task_id,
            'func': task_func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'type': task_type,
            'submitted_at': datetime.utcnow().isoformat(),
            'status': 'queued',
            'progress': 0,
            'result': None,
            'error': None
        }

        # Add to queue with priority
        self.task_queue.put((priority, task_id, task_data, time.time()))

        with self.lock:
            self.active_tasks[task_id] = task_data
            self.stats['total_queued'] += 1

        logger.info(f"Task {task_id} ({task_type}) submitted to background queue")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a background task.

        Args:
            task_id: Task identifier

        Returns:
            Task status dictionary or None if not found
        """
        with self.lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].copy()
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id].copy()
        return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued or running task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled, False otherwise
        """
        with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task['status'] == 'queued':
                    task['status'] = 'cancelled'
                    self.stats['total_failed'] += 1
                    logger.info(f"Task {task_id} cancelled")
                    return True
                elif task['status'] == 'running':
                    # For running tasks, we can only mark as cancelled
                    # The worker will check this flag
                    task['cancel_requested'] = True
                    logger.info(f"Cancel requested for running task {task_id}")
                    return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get background task statistics"""
        with self.lock:
            return {
                **self.stats,
                'active_tasks': len([t for t in self.active_tasks.values() if t['status'] == 'running']),
                'queued_tasks': len([t for t in self.active_tasks.values() if t['status'] == 'queued']),
                'completed_tasks': len(self.completed_tasks)
            }

    def _worker_loop(self):
        """Main worker thread loop"""
        logger.info(f"Background worker {threading.current_thread().name} started")

        while not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                try:
                    priority, task_id, task_data, queued_time = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Check if task was cancelled while waiting
                with self.lock:
                    if task_data.get('cancel_requested'):
                        task_data['status'] = 'cancelled'
                        self.stats['total_failed'] += 1
                        self.task_queue.task_done()
                        continue

                    task_data['status'] = 'running'
                    task_data['started_at'] = datetime.utcnow().isoformat()
                    self.stats['active_workers'] += 1

                # Execute task
                self._execute_task(task_data)

                # Mark queue task as done
                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")

        logger.info(f"Background worker {threading.current_thread().name} stopped")

    def _execute_task(self, task_data: Dict[str, Any]):
        """Execute a background task"""
        task_id = task_data['id']
        task_func = task_data['func']

        try:
            logger.info(f"Executing task {task_id}")

            # Execute the task function with Flask app context if available
            if self.app:
                with self.app.app_context():
                    result = task_func(*task_data['args'], **task_data['kwargs'])
            else:
                result = task_func(*task_data['args'], **task_data['kwargs'])

            # Update task status
            with self.lock:
                task_data['status'] = 'completed'
                task_data['result'] = result
                task_data['completed_at'] = datetime.utcnow().isoformat()
                task_data['progress'] = 100

                # Move to completed tasks (keep recent ones)
                self.completed_tasks[task_id] = task_data
                del self.active_tasks[task_id]

                self.stats['total_completed'] += 1
                self.stats['active_workers'] -= 1

            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")

            # Update task status on failure
            with self.lock:
                task_data['status'] = 'failed'
                task_data['error'] = str(e)
                task_data['failed_at'] = datetime.utcnow().isoformat()

                # Move to completed tasks
                self.completed_tasks[task_id] = task_data
                del self.active_tasks[task_id]

                self.stats['total_failed'] += 1
                self.stats['active_workers'] -= 1

        finally:
            # Clean up old completed tasks (keep last 1000)
            with self.lock:
                if len(self.completed_tasks) > 1000:
                    # Remove oldest completed tasks
                    oldest_tasks = sorted(
                        self.completed_tasks.items(),
                        key=lambda x: x[1].get('completed_at', ''),
                        reverse=True
                    )[1000:]
                    for task_id_to_remove, _ in oldest_tasks:
                        del self.completed_tasks[task_id_to_remove]

# Convenience functions for common background tasks

def process_batch_prediction(images_data: list, crop_type: str = None) -> Dict[str, Any]:
    """
    Background task for processing batch predictions.

    Args:
        images_data: List of image data dictionaries
        crop_type: Optional crop type hint

    Returns:
        Batch prediction results
    """
    from services.prediction_service import PredictionService

    prediction_service = PredictionService()
    results = []

    for image_data in images_data:
        try:
            # Process each image
            result = prediction_service.predict_disease(
                image_data['path'],
                crop_type,
                image_data.get('session_id')
            )
            results.append({
                'filename': image_data.get('filename', 'unknown'),
                'result': result,
                'success': True
            })
        except Exception as e:
            results.append({
                'filename': image_data.get('filename', 'unknown'),
                'error': str(e),
                'success': False
            })

    return {
        'total_processed': len(results),
        'successful': len([r for r in results if r['success']]),
        'results': results
    }

def generate_analytics_report(date_range: Dict[str, str]) -> Dict[str, Any]:
    """
    Background task for generating analytics reports.

    Args:
        date_range: Date range for the report

    Returns:
        Analytics report data
    """
    from services.session_service import SessionService

    session_service = SessionService()

    # Generate comprehensive analytics report
    report = {
        'generated_at': datetime.utcnow().isoformat(),
        'date_range': date_range,
        'summary': session_service.get_session_stats(),
        'charts': {
            'predictions_over_time': [],
            'disease_distribution': [],
            'accuracy_trends': []
        }
    }

    # This would include actual data processing logic
    # For now, return placeholder structure

    return report

def cleanup_expired_sessions() -> Dict[str, Any]:
    """
    Background task for cleaning up expired sessions and temporary files.

    Returns:
        Cleanup statistics
    """
    import os
    import shutil
    from datetime import datetime, timedelta

    cleanup_stats = {
        'sessions_cleaned': 0,
        'files_cleaned': 0,
        'space_freed': 0
    }

    # Clean up old temporary files (older than 24 hours)
    temp_dir = 'uploads'
    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Check file age
                    if time.time() - os.path.getmtime(file_path) > 86400:  # 24 hours
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleanup_stats['files_cleaned'] += 1
                        cleanup_stats['space_freed'] += size
                except Exception as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")

    # Clean up empty directories
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception:
                pass

    logger.info(f"Cleanup completed: {cleanup_stats}")
    return cleanup_stats