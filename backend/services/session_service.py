# services/session_service.py - Session management service
"""
Session Service - Manages prediction sessions and history.

Responsibilities:
- Store and retrieve prediction sessions
- Maintain prediction history
- Handle session cleanup and expiration
- Provide analytics data

Uses Redis for session storage and SQLite/PostgreSQL for persistent history.
"""

import json
import logging
import sqlite3
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import redis
from flask import current_app

logger = logging.getLogger(__name__)

class SessionService:
    """
    Service for managing prediction sessions and history.
    Provides caching and persistence for prediction data.
    """

    def __init__(self):
        """Initialize session service"""
        self.redis_client = None
        self.db_path = current_app.config.get('DATABASE_URL', 'sqlite:///sessions.db')
        self._init_redis()
        self._init_database()

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = current_app.config.get('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()  # Test connection
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}. Using in-memory storage.")
            self.redis_client = None

    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            # Extract path from sqlite:///path format
            if self.db_path.startswith('sqlite:///'):
                db_file = self.db_path[10:]
            else:
                db_file = 'sessions.db'

            # Ensure directory exists
            os.makedirs(os.path.dirname(db_file), exist_ok=True)

            self.db_connection = sqlite3.connect(db_file, check_same_thread=False)
            self._create_tables()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            self.db_connection = None

    def _create_tables(self):
        """Create necessary database tables"""
        if not self.db_connection:
            return

        try:
            cursor = self.db_connection.cursor()

            # Sessions table for persistent storage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    crop_type TEXT,
                    disease TEXT,
                    confidence REAL,
                    status TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    prediction_data TEXT
                )
            ''')

            # Analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    action TEXT,
                    timestamp TIMESTAMP,
                    metadata TEXT
                )
            ''')

            self.db_connection.commit()
            logger.info("Database tables created")

        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")

    def store_prediction(self, session_id: str, prediction_data: Dict):
        """
        Store prediction data in session.

        Args:
            session_id: Unique session identifier
            prediction_data: Prediction results and metadata
        """
        try:
            # Add timestamp and metadata
            session_data = {
                'data': prediction_data,
                'created_at': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat(),
                'version': '1.0'
            }

            # Store in Redis with expiration
            key = f"prediction:{session_id}"
            if self.redis_client:
                self.redis_client.setex(
                    key,
                    timedelta(hours=24),  # 24 hour expiration
                    json.dumps(session_data)
                )
            else:
                # Fallback to in-memory storage (not persistent)
                if not hasattr(self, '_memory_storage'):
                    self._memory_storage = {}
                self._memory_storage[key] = session_data

            # Also store in database for persistence
            self._store_in_database(session_id, prediction_data)

            logger.info(f"Stored prediction session: {session_id}")

        except Exception as e:
            logger.error(f"Failed to store prediction: {str(e)}")
            raise

    def _store_in_database(self, session_id: str, prediction_data: Dict):
        """Store prediction data in database"""
        if not self.db_connection:
            return

        try:
            cursor = self.db_connection.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO sessions
                (session_id, crop_type, disease, confidence, status, created_at, updated_at, prediction_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                prediction_data.get('crop_type', 'unknown'),
                prediction_data.get('predictions', [{}])[0].get('disease', 'unknown') if prediction_data.get('predictions') else 'unknown',
                prediction_data.get('confidence', 0.0),
                prediction_data.get('status', 'active'),
                datetime.utcnow(),
                datetime.utcnow(),
                json.dumps(prediction_data)
            ))

            self.db_connection.commit()

        except Exception as e:
            logger.error(f"Database storage failed: {str(e)}")

    def get_prediction(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve prediction data from session.

        Args:
            session_id: Session identifier

        Returns:
            Prediction data or None if not found
        """
        try:
            # Try Redis first
            key = f"prediction:{session_id}"

            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)

            # Fallback to database
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute('SELECT prediction_data FROM sessions WHERE session_id = ?', (session_id,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])

            # Final fallback to memory
            if hasattr(self, '_memory_storage'):
                return self._memory_storage.get(key)

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve prediction: {str(e)}")
            return None

    def update_prediction(self, session_id: str, updated_data: Dict):
        """
        Update existing prediction data.

        Args:
            session_id: Session identifier
            updated_data: Updated prediction data
        """
        try:
            existing = self.get_prediction(session_id)
            if existing:
                existing['data'].update(updated_data)
                existing['last_updated'] = datetime.utcnow().isoformat()
                self.store_prediction(session_id, existing['data'])

                # Log analytics event
                self._log_analytics(session_id, 'update', {'updated_fields': list(updated_data.keys())})

            else:
                logger.warning(f"Session not found for update: {session_id}")

        except Exception as e:
            logger.error(f"Failed to update prediction: {str(e)}")
            raise

    def get_prediction_history(self, limit: int = 10, offset: int = 0) -> List[Dict]:
        """
        Get prediction history with pagination.

        Args:
            limit: Maximum number of records to return
            offset: Pagination offset

        Returns:
            List of prediction records
        """
        if not self.db_connection:
            return []

        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT session_id, crop_type, disease, confidence, status, created_at, prediction_data
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))

            history = []
            for row in cursor.fetchall():
                prediction_data = json.loads(row[5]) if row[5] else {}
                history.append({
                    'session_id': row[0],
                    'crop_type': row[1],
                    'disease': row[2],
                    'confidence': row[3],
                    'status': row[4],
                    'timestamp': row[5],
                    'prediction_data': prediction_data
                })

            return history

        except Exception as e:
            logger.error(f"Failed to retrieve history: {str(e)}")
            return []

    def cleanup_expired_sessions(self):
        """
        Clean up expired sessions.
        Should be called periodically by a background task.
        """
        try:
            # Redis handles expiration automatically
            # Clean up old database records (older than 30 days)
            if self.db_connection:
                cursor = self.db_connection.cursor()
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)

                cursor.execute('''
                    DELETE FROM sessions
                    WHERE created_at < ?
                ''', (thirty_days_ago,))

                deleted_count = cursor.rowcount
                self.db_connection.commit()

                logger.info(f"Cleaned up {deleted_count} expired sessions")

        except Exception as e:
            logger.error(f"Session cleanup failed: {str(e)}")

    def get_session_stats(self) -> Dict:
        """
        Get session statistics for analytics.

        Returns:
            Dictionary with session statistics
        """
        if not self.db_connection:
            return {}

        try:
            cursor = self.db_connection.cursor()

            # Get total sessions
            cursor.execute('SELECT COUNT(*) FROM sessions')
            total_sessions = cursor.fetchone()[0]

            # Get active sessions (created in last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            cursor.execute('SELECT COUNT(*) FROM sessions WHERE created_at > ?', (yesterday,))
            active_sessions = cursor.fetchone()[0]

            # Get average confidence
            cursor.execute('SELECT AVG(confidence) FROM sessions WHERE confidence > 0')
            avg_confidence = cursor.fetchone()[0] or 0.0

            # Get popular crops
            cursor.execute('''
                SELECT crop_type, COUNT(*) as count
                FROM sessions
                WHERE crop_type != 'unknown'
                GROUP BY crop_type
                ORDER BY count DESC
                LIMIT 5
            ''')
            popular_crops = [{'crop': row[0], 'count': row[1]} for row in cursor.fetchall()]

            # Get popular diseases
            cursor.execute('''
                SELECT disease, COUNT(*) as count
                FROM sessions
                WHERE disease != 'unknown'
                GROUP BY disease
                ORDER BY count DESC
                LIMIT 5
            ''')
            popular_diseases = [{'disease': row[0], 'count': row[1]} for row in cursor.fetchall()]

            stats = {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'avg_confidence': round(avg_confidence, 3),
                'popular_crops': popular_crops,
                'popular_diseases': popular_diseases
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get session stats: {str(e)}")
            return {}

    def _log_analytics(self, session_id: str, action: str, metadata: Dict = None):
        """Log analytics event"""
        if not self.db_connection:
            return

        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO analytics (session_id, action, timestamp, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                session_id,
                action,
                datetime.utcnow(),
                json.dumps(metadata or {})
            ))
            self.db_connection.commit()

        except Exception as e:
            logger.error(f"Analytics logging failed: {str(e)}")

    def export_sessions(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """
        Export session data for analysis.

        Args:
            start_date: Start date for export
            end_date: End date for export

        Returns:
            List of session data for export
        """
        if not self.db_connection:
            return []

        try:
            cursor = self.db_connection.cursor()

            query = 'SELECT * FROM sessions WHERE 1=1'
            params = []

            if start_date:
                query += ' AND created_at >= ?'
                params.append(start_date)

            if end_date:
                query += ' AND created_at <= ?'
                params.append(end_date)

            query += ' ORDER BY created_at DESC'

            cursor.execute(query, params)

            export_data = []
            for row in cursor.fetchall():
                export_data.append({
                    'session_id': row[0],
                    'crop_type': row[1],
                    'disease': row[2],
                    'confidence': row[3],
                    'status': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'prediction_data': json.loads(row[7]) if row[7] else {}
                })

            return export_data

        except Exception as e:
            logger.error(f"Session export failed: {str(e)}")
            return []

    def get_prediction(self, key: str):
        """
        Retrieve prediction data by key.

        Args:
            key: Prediction key

        Returns:
            Prediction data or None if not found
        """
        try:
            if hasattr(self, '_memory_storage'):
                return self._memory_storage.get(key)

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve prediction: {str(e)}")
            return None

    def update_prediction(self, session_id: str, updated_data: Dict):
        """
        Update existing prediction data.

        Args:
            session_id: Session identifier
            updated_data: Updated prediction data
        """
        try:
            existing = self.get_prediction(session_id)
            if existing:
                existing['data'].update(updated_data)
                existing['last_updated'] = datetime.utcnow().isoformat()
                self.store_prediction(session_id, existing['data'])
            else:
                logger.warning(f"Session not found for update: {session_id}")

        except Exception as e:
            logger.error(f"Failed to update prediction: {str(e)}")
            raise

    def get_prediction_history(self, limit: int = 10, offset: int = 0) -> List[Dict]:
        """
        Get prediction history with pagination.

        Args:
            limit: Maximum number of records to return
            offset: Pagination offset

        Returns:
            List of prediction records
        """
        # In production, this would query a database
        # For now, return mock data
        try:
            # This is a simplified implementation
            # Real implementation would query persistent storage
            history = []

            # Mock history data
            for i in range(min(limit, 5)):  # Mock 5 records max
                history.append({
                    'session_id': f'mock_session_{i}',
                    'crop_type': 'tomato',
                    'disease': 'early_blight',
                    'confidence': 0.85,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'completed'
                })

            return history

        except Exception as e:
            logger.error(f"Failed to retrieve history: {str(e)}")
            return []

    def cleanup_expired_sessions(self):
        """
        Clean up expired sessions.
        Should be called periodically by a background task.
        """
        try:
            # Redis handles expiration automatically
            # For database cleanup, implement here
            logger.info("Session cleanup completed")
        except Exception as e:
            logger.error(f"Session cleanup failed: {str(e)}")

    def get_session_stats(self) -> Dict:
        """
        Get session statistics for analytics.

        Returns:
            Dictionary with session statistics
        """
        try:
            stats = {
                'total_sessions': 0,  # Would query database
                'active_sessions': 0,
                'avg_confidence': 0.0,
                'popular_crops': [],
                'popular_diseases': []
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get session stats: {str(e)}")
            return {}