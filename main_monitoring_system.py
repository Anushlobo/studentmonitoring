#main_monitoring_system.py
import cv2
import numpy as np
import time
import logging
import threading
from datetime import datetime, timedelta
import uuid
import os
from typing import Dict, List, Optional

from config import (
    FRAME_INTERVAL, VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    ATTENTION_WINDOW_MINUTES, ATTENDANCE_WINDOW_MINUTES, CHEATING_DETECTION_ENABLED,
    SAVE_ANALYSIS_FRAMES, ANALYSIS_FRAMES_PATH, SAVE_ATTENDANCE_REPORTS,
    ATTENDANCE_REPORTS_PATH, LOG_LEVEL, LOG_FILE
)
from database import MonitoringDatabase
from face_recognition_module import FaceRecognitionModule
from attention_analysis import AttentionAnalysisModule
from student_detection import StudentDetectionModule

class StudentMonitoringSystem:
    def __init__(self):
        """Initialize the student monitoring system"""
        # Setup logging
        self.setup_logging()
        
        # Initialize modules
        self.database = MonitoringDatabase()
        self.face_recognition = FaceRecognitionModule()
        self.attention_analysis = AttentionAnalysisModule()
        self.student_detection = StudentDetectionModule()
        
        # Video capture
        self.cap = None
        self.frame_count = 0
        self.last_processed_frame = 0
        
        # Session management
        self.session_id = None
        self.session_start_time = None
        self.exam_mode = False
        
        # Analysis tracking
        self.student_analysis_history = {}
        self.attendance_tracking = {}
        self.frame_analysis_results = []
        
        # Threading
        self.is_running = False
        self.monitoring_thread = None
        
        logging.info("Student Monitoring System initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    def start_session(self, session_type="regular", exam_mode=False, expected_students=0):
        """Start a new monitoring session"""
        try:
            self.session_id = str(uuid.uuid4())
            self.session_start_time = datetime.now()
            self.exam_mode = exam_mode
            
            # Create session in database
            self.database.create_session(
                self.session_id, 
                session_type, 
                expected_students, 
                exam_mode
            )
            
            logging.info(f"Started monitoring session: {self.session_id}")
            logging.info(f"Session type: {session_type}, Exam mode: {exam_mode}")
            
            return self.session_id
            
        except Exception as e:
            logging.error(f"Error starting session: {e}")
            return None
    
    def end_session(self):
        """End the current monitoring session"""
        try:
            if self.session_id:
                self.database.end_session(self.session_id)
                logging.info(f"Ended monitoring session: {self.session_id}")
                
                # Generate final reports
                self.generate_session_reports()
                
                self.session_id = None
                self.session_start_time = None
                self.exam_mode = False
                
        except Exception as e:
            logging.error(f"Error ending session: {e}")
    
    def initialize_video_capture(self):
        """Initialize video capture"""
        try:
            self.cap = cv2.VideoCapture(VIDEO_SOURCE)
            
            if not self.cap.isOpened():
                logging.error("Error: Could not open video source")
                return False
            
            # Set video properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)
            
            logging.info(f"Video capture initialized: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing video capture: {e}")
            return False
    
    def process_frame(self, frame, bypass_interval=False):
        """Process a single frame for analysis"""
        try:
            self.frame_count += 1
            
            # Check if we should process this frame (every 30 seconds)
            # Skip interval check if bypass_interval is True (for video processing)
            if not bypass_interval and self.frame_count - self.last_processed_frame < FRAME_INTERVAL * FPS:
                return None
            
            self.last_processed_frame = self.frame_count
            
            logging.info(f"Processing frame {self.frame_count} at {datetime.now()}")
            
            # Step 1: Detect students using YOLO
            student_detections = self.student_detection.detect_people(frame)
            student_detections = self.student_detection.filter_students(student_detections)
            
            if not student_detections:
                logging.info("No students detected in frame")
                return None
            
            # Step 2: Recognize faces for each detected student
            frame_results = []
            
            for i, detection in enumerate(student_detections):
                # Get ROI for face recognition
                roi = self.student_detection.get_detection_roi(detection)
                x1, y1, x2, y2 = roi
                
                # Extract face region
                face_region = frame[y1:y2, x1:x2]
                
                if face_region.size == 0:
                    continue
                
                # Recognize student
                recognition_results = self.face_recognition.recognize_students(face_region)
                
                if not recognition_results:
                    # Unknown student
                    student_id = f"unknown_{i}"
                    confidence = 0.0
                else:
                    # Use the first recognized face
                    recognition_result = recognition_results[0]
                    student_id = recognition_result['student_id']
                    confidence = recognition_result['confidence']
                
                # Step 3: Analyze attention
                # Convert bbox format from (x1, y1, x2, y2) to (top, right, bottom, left)
                x1, y1, x2, y2 = detection['bbox']
                face_location = (y1, x2, y2, x1)  # (top, right, bottom, left)
                attention_data = self.attention_analysis.analyze_attention(frame, face_location)
                
                # Step 4: Detect cheating (if in exam mode)
                cheating_data = self.attention_analysis.detect_cheating(attention_data, self.exam_mode)
                
                # Combine all results
                result = {
                    'frame_number': self.frame_count,
                    'timestamp': datetime.now(),
                    'student_id': student_id,
                    'recognition_confidence': confidence,
                    'detection_bbox': detection['bbox'],
                    'attention_score': attention_data.get('attention_score', 0.0),
                    'distraction_score': attention_data.get('distraction_score', 1.0),
                    'head_pose_x': attention_data.get('head_pose', {}).get('pitch', 0.0),
                    'head_pose_y': attention_data.get('head_pose', {}).get('yaw', 0.0),
                    'head_pose_z': attention_data.get('head_pose', {}).get('roll', 0.0),
                    'eye_aspect_ratio': attention_data.get('eye_aspect_ratio', 0.0),
                    'is_sleeping': attention_data.get('is_sleeping', False),
                    'phone_detected': attention_data.get('phone_detected', False),
                    'cheating_probability': cheating_data.get('cheating_probability', 0.0),
                    'session_id': self.session_id,
                    'face_detected': attention_data.get('face_detected', False)
                }
                
                frame_results.append(result)
                
                # Log to database
                self.database.log_monitoring_data(result)
                
                # Track attendance
                if student_id != "Unknown" and confidence > 0.5:
                    self.track_attendance(student_id, confidence)
                
                # Update analysis history
                if student_id not in self.student_analysis_history:
                    self.student_analysis_history[student_id] = []
                self.student_analysis_history[student_id].append(result)
            
            # Save analysis frame if enabled
            if SAVE_ANALYSIS_FRAMES:
                self.save_analysis_frame(frame, frame_results)
            
            self.frame_analysis_results.append({
                'frame_number': self.frame_count,
                'timestamp': datetime.now(),
                'results': frame_results,
                'total_students': len(frame_results)
            })
            
            return frame_results
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None
    
    def track_attendance(self, student_id, confidence):
        """Track student attendance"""
        try:
            current_time = datetime.now()
            
            # Check if student is already marked present recently
            if student_id in self.attendance_tracking:
                last_seen = self.attendance_tracking[student_id]['last_seen']
                if (current_time - last_seen).total_seconds() < ATTENDANCE_WINDOW_MINUTES * 60:
                    # Update last seen time
                    self.attendance_tracking[student_id]['last_seen'] = current_time
                    return
            
            # Mark student as present
            self.attendance_tracking[student_id] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'confidence': confidence
            }
            
            # Log attendance to database
            self.database.log_attendance(
                student_id, 
                self.session_id, 
                'present', 
                confidence
            )
            
            logging.info(f"Student {student_id} marked present with confidence {confidence}")
            
        except Exception as e:
            logging.error(f"Error tracking attendance: {e}")
    
    def save_analysis_frame(self, frame, results):
        """Save frame with analysis annotations"""
        try:
            annotated_frame = frame.copy()
            
            for result in results:
                bbox = result['detection_bbox']
                student_id = result['student_id']
                attention_score = result['attention_score']
                cheating_prob = result['cheating_probability']
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if attention_score > 0.7 else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add text labels
                label = f"{student_id}: {attention_score:.2f}"
                if cheating_prob > 0.5:
                    label += f" (CHEAT: {cheating_prob:.2f})"
                
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{self.frame_count}_{timestamp}.jpg"
            filepath = os.path.join(ANALYSIS_FRAMES_PATH, filename)
            
            cv2.imwrite(filepath, annotated_frame)
            
        except Exception as e:
            logging.error(f"Error saving analysis frame: {e}")
    
    def generate_session_reports(self):
        """Generate reports for the session"""
        try:
            if not self.session_id:
                return
            
            # Generate attendance report
            attendance_report = self.database.get_attendance_report(self.session_id)
            
            if SAVE_ATTENDANCE_REPORTS and not attendance_report.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"attendance_report_{self.session_id}_{timestamp}.csv"
                filepath = os.path.join(ATTENDANCE_REPORTS_PATH, filename)
                
                attendance_report.to_csv(filepath, index=False)
                logging.info(f"Attendance report saved: {filepath}")
            
            # Generate attention analysis report
            attention_report = self.database.get_attention_analysis(self.session_id)
            
            if not attention_report.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"attention_report_{self.session_id}_{timestamp}.csv"
                filepath = os.path.join(ATTENDANCE_REPORTS_PATH, filename)
                
                attention_report.to_csv(filepath, index=False)
                logging.info(f"Attention report saved: {filepath}")
            
            # Generate summary statistics
            self.generate_summary_statistics()
            
        except Exception as e:
            logging.error(f"Error generating session reports: {e}")
    
    def generate_summary_statistics(self):
        """Generate summary statistics for the session"""
        try:
            if not self.frame_analysis_results:
                return
            
            total_frames = len(self.frame_analysis_results)
            total_students_detected = len(self.attendance_tracking)
            
            # Calculate average attention scores
            all_attention_scores = []
            all_cheating_probs = []
            
            for frame_result in self.frame_analysis_results:
                for result in frame_result['results']:
                    all_attention_scores.append(result['attention_score'])
                    all_cheating_probs.append(result['cheating_probability'])
            
            avg_attention = sum(all_attention_scores) / len(all_attention_scores) if all_attention_scores else 0
            avg_cheating_prob = sum(all_cheating_probs) / len(all_cheating_probs) if all_cheating_probs else 0
            
            # Generate summary
            summary = {
                'session_id': self.session_id,
                'session_duration': str(datetime.now() - self.session_start_time),
                'total_frames_analyzed': total_frames,
                'total_students_detected': total_students_detected,
                'average_attention_score': avg_attention,
                'average_cheating_probability': avg_cheating_prob,
                'exam_mode': self.exam_mode
            }
            
            logging.info("Session Summary:")
            for key, value in summary.items():
                logging.info(f"  {key}: {value}")
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating summary statistics: {e}")
            return None
    
    def start_monitoring(self):
        """Start the monitoring process"""
        try:
            if not self.initialize_video_capture():
                return False
            
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.start()
            
            logging.info("Monitoring started")
            return True
            
        except Exception as e:
            logging.error(f"Error starting monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logging.error("Error reading frame from video source")
                    break
                
                # Process frame
                self.process_frame(frame)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            logging.error(f"Error in monitoring loop: {e}")
        finally:
            if self.cap:
                self.cap.release()
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        try:
            self.is_running = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            if self.cap:
                self.cap.release()
            
            # End session
            self.end_session()
            
            logging.info("Monitoring stopped")
            
        except Exception as e:
            logging.error(f"Error stopping monitoring: {e}")
    
    def get_current_status(self):
        """Get current system status"""
        try:
            status = {
                'is_running': self.is_running,
                'session_id': self.session_id,
                'frame_count': self.frame_count,
                'total_students_tracked': len(self.attendance_tracking),
                'session_duration': str(datetime.now() - self.session_start_time) if self.session_start_time else "N/A",
                'exam_mode': self.exam_mode
            }
            
            return status
            
        except Exception as e:
            logging.error(f"Error getting status: {e}")
            return {}
    
    def get_student_analysis(self, student_id):
        """Get analysis data for a specific student"""
        try:
            if student_id in self.student_analysis_history:
                return self.student_analysis_history[student_id]
            else:
                return []
                
        except Exception as e:
            logging.error(f"Error getting student analysis: {e}")
            return []

