import sqlite3
import pandas as pd
import csv
import os
from datetime import datetime
import logging
from config import DATABASE_PATH, CSV_LOG_PATH

class MonitoringDatabase:
    def __init__(self):
        """Initialize the monitoring database"""
        self.db_path = DATABASE_PATH
        self.csv_path = CSV_LOG_PATH
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create students table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create monitoring_logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    student_id TEXT,
                    face_detected BOOLEAN,
                    attention_score REAL,
                    distraction_score REAL,
                    head_pose_x REAL,
                    head_pose_y REAL,
                    head_pose_z REAL,
                    eye_aspect_ratio REAL,
                    is_sleeping BOOLEAN,
                    phone_detected BOOLEAN,
                    cheating_probability REAL,
                    frame_number INTEGER,
                    session_id TEXT
                )
            ''')
            
            # Create attendance_logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    student_id TEXT,
                    session_id TEXT,
                    attendance_status TEXT,
                    confidence_score REAL
                )
            ''')
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    session_type TEXT,
                    total_students INTEGER,
                    exam_mode BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            
            # Initialize CSV file with headers
            self.init_csv_log()
            
        except Exception as e:
            logging.error(f"Database initialization error: {e}")
    
    def init_csv_log(self):
        """Initialize CSV log file with headers"""
        if not os.path.exists(self.csv_path):
            headers = [
                'timestamp', 'student_id', 'name', 'face_detected', 
                'attention_score', 'distraction_score', 'head_pose_x', 
                'head_pose_y', 'head_pose_z', 'eye_aspect_ratio', 
                'is_sleeping', 'phone_detected', 'cheating_probability',
                'frame_number', 'session_id'
            ]
            
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
    
    def add_student(self, student_id, name):
        """Add a new student to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO students (student_id, name)
                VALUES (?, ?)
            ''', (student_id, name))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error adding student: {e}")
            return False
    
    def log_monitoring_data(self, monitoring_data):
        """Log monitoring data to database and CSV"""
        try:
            # Log to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO monitoring_logs 
                (student_id, face_detected, attention_score, distraction_score,
                 head_pose_x, head_pose_y, head_pose_z, eye_aspect_ratio,
                 is_sleeping, phone_detected, cheating_probability, 
                 frame_number, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                monitoring_data.get('student_id'),
                monitoring_data.get('face_detected', False),
                monitoring_data.get('attention_score', 0.0),
                monitoring_data.get('distraction_score', 0.0),
                monitoring_data.get('head_pose_x', 0.0),
                monitoring_data.get('head_pose_y', 0.0),
                monitoring_data.get('head_pose_z', 0.0),
                monitoring_data.get('eye_aspect_ratio', 0.0),
                monitoring_data.get('is_sleeping', False),
                monitoring_data.get('phone_detected', False),
                monitoring_data.get('cheating_probability', 0.0),
                monitoring_data.get('frame_number', 0),
                monitoring_data.get('session_id', '')
            ))
            
            conn.commit()
            conn.close()
            
            # Log to CSV
            self.log_to_csv(monitoring_data)
            
            return True
            
        except Exception as e:
            logging.error(f"Error logging monitoring data: {e}")
            return False
    
    def log_to_csv(self, monitoring_data):
        """Log monitoring data to CSV file"""
        try:
            # Get student name
            student_name = self.get_student_name(monitoring_data.get('student_id', ''))
            
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                monitoring_data.get('student_id', ''),
                student_name,
                monitoring_data.get('face_detected', False),
                monitoring_data.get('attention_score', 0.0),
                monitoring_data.get('distraction_score', 0.0),
                monitoring_data.get('head_pose_x', 0.0),
                monitoring_data.get('head_pose_y', 0.0),
                monitoring_data.get('head_pose_z', 0.0),
                monitoring_data.get('eye_aspect_ratio', 0.0),
                monitoring_data.get('is_sleeping', False),
                monitoring_data.get('phone_detected', False),
                monitoring_data.get('cheating_probability', 0.0),
                monitoring_data.get('frame_number', 0),
                monitoring_data.get('session_id', '')
            ]
            
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)
                
        except Exception as e:
            logging.error(f"Error logging to CSV: {e}")
    
    def get_student_name(self, student_id):
        """Get student name from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
            result = cursor.fetchone()
            
            conn.close()
            
            return result[0] if result else 'Unknown'
            
        except Exception as e:
            logging.error(f"Error getting student name: {e}")
            return 'Unknown'
    
    def log_attendance(self, student_id, session_id, attendance_status, confidence_score):
        """Log attendance data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO attendance_logs 
                (student_id, session_id, attendance_status, confidence_score)
                VALUES (?, ?, ?, ?)
            ''', (student_id, session_id, attendance_status, confidence_score))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error logging attendance: {e}")
            return False
    
    def create_session(self, session_id, session_type, total_students, exam_mode=False):
        """Create a new monitoring session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions 
                (session_id, session_type, total_students, exam_mode)
                VALUES (?, ?, ?, ?)
            ''', (session_id, session_type, total_students, exam_mode))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error creating session: {e}")
            return False
    
    def end_session(self, session_id):
        """End a monitoring session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions 
                SET end_time = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            ''', (session_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error ending session: {e}")
            return False
    
    def get_attendance_report(self, session_id):
        """Get attendance report for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    al.student_id,
                    s.name,
                    al.attendance_status,
                    al.confidence_score,
                    al.timestamp
                FROM attendance_logs al
                LEFT JOIN students s ON al.student_id = s.student_id
                WHERE al.session_id = ?
                ORDER BY al.timestamp
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting attendance report: {e}")
            return pd.DataFrame()
    
    def get_attention_analysis(self, session_id, student_id=None):
        """Get attention analysis for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if student_id:
                query = '''
                    SELECT 
                        timestamp,
                        attention_score,
                        distraction_score,
                        head_pose_x,
                        head_pose_y,
                        head_pose_z,
                        eye_aspect_ratio,
                        is_sleeping,
                        phone_detected,
                        cheating_probability
                    FROM monitoring_logs
                    WHERE session_id = ? AND student_id = ?
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn, params=(session_id, student_id))
            else:
                query = '''
                    SELECT 
                        student_id,
                        timestamp,
                        attention_score,
                        distraction_score,
                        head_pose_x,
                        head_pose_y,
                        head_pose_z,
                        eye_aspect_ratio,
                        is_sleeping,
                        phone_detected,
                        cheating_probability
                    FROM monitoring_logs
                    WHERE session_id = ?
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn, params=(session_id,))
            
            conn.close()
            return df
            
        except Exception as e:
            logging.error(f"Error getting attention analysis: {e}")
            return pd.DataFrame()

