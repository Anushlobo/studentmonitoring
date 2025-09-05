#student_detection.py
import cv2
import numpy as np
import logging
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from config import YOLO_MODEL_PATH, PERSON_CONFIDENCE_THRESHOLD

class StudentDetectionModule:
    def __init__(self):
        """Initialize the student detection module using YOLO"""
        try:
            # Load YOLO model
            self.model = YOLO(YOLO_MODEL_PATH)
            logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            self.model = None
    
    def detect_people(self, frame):
        """Detect people in the frame using YOLO"""
        try:
            if self.model is None:
                logging.error("YOLO model not loaded")
                return []
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only process person class (class_id 0 for COCO dataset)
                        if class_id == 0 and confidence > PERSON_CONFIDENCE_THRESHOLD:
                            detection = {
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': 'person'
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logging.error(f"Error detecting people: {e}")
            return []
    
    def filter_students(self, detections, min_size=50):
        """Filter detections to identify likely students based on size and position"""
        try:
            student_detections = []
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                
                # Calculate detection size
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Filter based on size (students should be reasonably sized)
                if area >= min_size * min_size:
                    # Calculate aspect ratio
                    aspect_ratio = height / width if width > 0 else 0
                    
                    # Students typically have aspect ratio > 1.5 (taller than wide)
                    if aspect_ratio > 1.2:
                        student_detections.append(detection)
            
            return student_detections
            
        except Exception as e:
            logging.error(f"Error filtering students: {e}")
            return []
    
    def estimate_student_count(self, detections):
        """Estimate the number of students based on detections"""
        try:
            # Count unique detections (with some overlap handling)
            if not detections:
                return 0
            
            # Simple overlap-based counting
            unique_detections = []
            
            for detection in detections:
                is_unique = True
                x1, y1, x2, y2 = detection['bbox']
                
                for existing in unique_detections:
                    ex1, ey1, ex2, ey2 = existing['bbox']
                    
                    # Calculate overlap
                    overlap_x = max(0, min(x2, ex2) - max(x1, ex1))
                    overlap_y = max(0, min(y2, ey2) - max(y1, ey1))
                    overlap_area = overlap_x * overlap_y
                    
                    detection_area = (x2 - x1) * (y2 - y1)
                    existing_area = (ex2 - ex1) * (ey2 - ey1)
                    
                    # If overlap is more than 50% of either detection, consider it the same person
                    if overlap_area > 0.5 * min(detection_area, existing_area):
                        is_unique = False
                        break
                
                if is_unique:
                    unique_detections.append(detection)
            
            return len(unique_detections)
            
        except Exception as e:
            logging.error(f"Error estimating student count: {e}")
            return 0
    
    def get_detection_roi(self, detection):
        """Get region of interest for a detection"""
        try:
            x1, y1, x2, y2 = detection['bbox']
            
            # Expand ROI slightly to include more context
            margin = 20
            roi_x1 = max(0, x1 - margin)
            roi_y1 = max(0, y1 - margin)
            roi_x2 = x2 + margin
            roi_y2 = y2 + margin
            
            return (roi_x1, roi_y1, roi_x2, roi_y2)
            
        except Exception as e:
            logging.error(f"Error getting detection ROI: {e}")
            return detection['bbox']
    
    def analyze_classroom_density(self, detections, frame_shape):
        """Analyze classroom density and seating patterns"""
        try:
            if not detections:
                return {
                    'total_students': 0,
                    'density_score': 0.0,
                    'seating_pattern': 'unknown',
                    'classroom_utilization': 0.0
                }
            
            frame_height, frame_width = frame_shape[:2]
            frame_area = frame_height * frame_width
            
            # Calculate total area occupied by students
            total_student_area = 0
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                area = (x2 - x1) * (y2 - y1)
                total_student_area += area
            
            # Calculate density score
            density_score = total_student_area / frame_area if frame_area > 0 else 0
            
            # Determine seating pattern based on spatial distribution
            if len(detections) > 1:
                # Calculate center points of all detections
                centers = []
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    centers.append((center_x, center_y))
                
                # Analyze spatial distribution
                if len(centers) >= 2:
                    # Calculate average distance between centers
                    total_distance = 0
                    count = 0
                    
                    for i in range(len(centers)):
                        for j in range(i + 1, len(centers)):
                            dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                         (centers[i][1] - centers[j][1])**2)
                            total_distance += dist
                            count += 1
                    
                    avg_distance = total_distance / count if count > 0 else 0
                    
                    # Determine seating pattern
                    if avg_distance < frame_width * 0.1:
                        seating_pattern = 'crowded'
                    elif avg_distance < frame_width * 0.2:
                        seating_pattern = 'normal'
                    else:
                        seating_pattern = 'spread_out'
                else:
                    seating_pattern = 'single_student'
            else:
                seating_pattern = 'single_student'
            
            # Calculate classroom utilization
            classroom_utilization = len(detections) / 50.0  # Assuming max 50 students
            classroom_utilization = min(1.0, classroom_utilization)
            
            return {
                'total_students': len(detections),
                'density_score': density_score,
                'seating_pattern': seating_pattern,
                'classroom_utilization': classroom_utilization
            }
            
        except Exception as e:
            logging.error(f"Error analyzing classroom density: {e}")
            return {
                'total_students': 0,
                'density_score': 0.0,
                'seating_pattern': 'unknown',
                'classroom_utilization': 0.0
            }
    
    def detect_movement(self, current_detections, previous_detections, threshold=30):
        """Detect movement between frames"""
        try:
            if not previous_detections:
                return {'movement_detected': False, 'movement_score': 0.0}
            
            movement_score = 0.0
            total_movement = 0
            
            for current in current_detections:
                current_center = (
                    (current['bbox'][0] + current['bbox'][2]) / 2,
                    (current['bbox'][1] + current['bbox'][3]) / 2
                )
                
                min_distance = float('inf')
                
                for previous in previous_detections:
                    previous_center = (
                        (previous['bbox'][0] + previous['bbox'][2]) / 2,
                        (previous['bbox'][1] + previous['bbox'][3]) / 2
                    )
                    
                    distance = np.sqrt(
                        (current_center[0] - previous_center[0])**2 +
                        (current_center[1] - previous_center[1])**2
                    )
                    
                    min_distance = min(min_distance, distance)
                
                if min_distance < float('inf'):
                    total_movement += min_distance
            
            # Calculate average movement
            if current_detections:
                avg_movement = total_movement / len(current_detections)
                movement_score = min(1.0, avg_movement / threshold)
            
            movement_detected = movement_score > 0.1
            
            return {
                'movement_detected': movement_detected,
                'movement_score': movement_score
            }
            
        except Exception as e:
            logging.error(f"Error detecting movement: {e}")
            return {'movement_detected': False, 'movement_score': 0.0}
    
    def get_detection_summary(self, detections):
        """Get a summary of detections"""
        try:
            if not detections:
                return {
                    'count': 0,
                    'avg_confidence': 0.0,
                    'size_distribution': 'none',
                    'detection_quality': 'poor'
                }
            
            # Calculate average confidence
            confidences = [d['confidence'] for d in detections]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Analyze size distribution
            areas = []
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)
            
            if areas:
                avg_area = sum(areas) / len(areas)
                if avg_area < 5000:
                    size_distribution = 'small'
                elif avg_area < 15000:
                    size_distribution = 'medium'
                else:
                    size_distribution = 'large'
            else:
                size_distribution = 'none'
            
            # Determine detection quality
            if avg_confidence > 0.8 and len(detections) > 0:
                detection_quality = 'excellent'
            elif avg_confidence > 0.6:
                detection_quality = 'good'
            elif avg_confidence > 0.4:
                detection_quality = 'fair'
            else:
                detection_quality = 'poor'
            
            return {
                'count': len(detections),
                'avg_confidence': avg_confidence,
                'size_distribution': size_distribution,
                'detection_quality': detection_quality
            }
            
        except Exception as e:
            logging.error(f"Error getting detection summary: {e}")
            return {
                'count': 0,
                'avg_confidence': 0.0,
                'size_distribution': 'none',
                'detection_quality': 'poor'
            }

