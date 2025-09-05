import cv2
import mediapipe as mp
import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional
from config import (
    ATTENTION_THRESHOLD, DISTRACTION_THRESHOLD, EYE_ASPECT_RATIO_THRESHOLD,
    HEAD_POSE_THRESHOLD, SLEEPING_DETECTION_THRESHOLD, PHONE_DETECTION_CONFIDENCE
)

class AttentionAnalysisModule:
    def __init__(self):
        """Initialize the attention analysis module"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,  # Support multiple faces
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Pose for body posture
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Hands for phone detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=4,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Eye aspect ratio calculation points
        self.EYE_AR_THRESH = EYE_ASPECT_RATIO_THRESHOLD
        self.EYE_AR_CONSEC_FRAMES = 3
        
        # Face mesh landmarks for different features
        self.FACE_LANDMARKS = {
            'left_eye': [362, 385, 387, 263, 373, 380],
            'right_eye': [33, 160, 158, 133, 153, 144],
            'nose': [1, 2, 3, 4, 5, 6],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        }
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate the eye aspect ratio"""
        try:
            # Vertical distances
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Horizontal distance
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Eye aspect ratio
            ear = (A + B) / (2.0 * C)
            
            return ear
        except Exception as e:
            logging.error(f"Error calculating eye aspect ratio: {e}")
            return 0.0
    
    def calculate_head_pose(self, face_landmarks):
        """Calculate head pose angles (pitch, yaw, roll)"""
        try:
            # Get key facial landmarks for head pose estimation
            # Nose tip, left eye corner, right eye corner
            nose_tip = face_landmarks[1]
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            
            # Calculate head pose using 2D landmarks
            # This is a simplified approach - for more accurate results,
            # you would need 3D landmarks or a dedicated head pose model
            
            # Calculate roll (head tilt)
            eye_center = (left_eye + right_eye) / 2
            roll_angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            roll_degrees = math.degrees(roll_angle)
            
            # Calculate yaw (head turn left/right)
            eye_distance = np.linalg.norm(right_eye - left_eye)
            nose_to_eye_center = np.linalg.norm(nose_tip - eye_center)
            yaw_angle = math.asin(nose_to_eye_center / eye_distance)
            yaw_degrees = math.degrees(yaw_angle)
            
            # Simplified pitch calculation
            pitch_degrees = 0.0  # Would need more landmarks for accurate pitch
            
            return {
                'pitch': pitch_degrees,
                'yaw': yaw_degrees,
                'roll': roll_degrees
            }
            
        except Exception as e:
            logging.error(f"Error calculating head pose: {e}")
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
    
    def detect_sleeping(self, left_ear, right_ear, head_pose):
        """Detect if person is sleeping based on eye closure and head pose"""
        try:
            # Check if eyes are closed
            eyes_closed = (left_ear < self.EYE_AR_THRESH and right_ear < self.EYE_AR_THRESH)
            
            # Check head pose for sleeping position (head tilted down)
            head_tilted_down = abs(head_pose['pitch']) > 30 or abs(head_pose['roll']) > 45
            
            # Calculate sleeping probability
            sleeping_score = 0.0
            
            if eyes_closed:
                sleeping_score += 0.6
            
            if head_tilted_down:
                sleeping_score += 0.4
            
            is_sleeping = sleeping_score > SLEEPING_DETECTION_THRESHOLD
            
            return {
                'is_sleeping': is_sleeping,
                'sleeping_score': sleeping_score,
                'eyes_closed': eyes_closed,
                'head_tilted_down': head_tilted_down
            }
            
        except Exception as e:
            logging.error(f"Error detecting sleeping: {e}")
            return {'is_sleeping': False, 'sleeping_score': 0.0, 'eyes_closed': False, 'head_tilted_down': False}
    
    def detect_phone_usage(self, frame, face_location):
        """Detect phone usage using hand detection and object detection"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            hand_results = self.hands.process(rgb_frame)
            
            if not hand_results.multi_hand_landmarks:
                return {'phone_detected': False, 'confidence': 0.0}
            
            # Check if hands are near face area (potential phone usage)
            face_top, face_right, face_bottom, face_left = face_location
            
            phone_detected = False
            max_confidence = 0.0
            
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get hand center
                hand_center_x = sum([landmark.x * frame.shape[1] for landmark in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
                hand_center_y = sum([landmark.y * frame.shape[0] for landmark in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
                
                # Check if hand is near face
                face_center_x = (face_left + face_right) / 2
                face_center_y = (face_top + face_bottom) / 2
                
                distance = math.sqrt((hand_center_x - face_center_x)**2 + (hand_center_y - face_center_y)**2)
                face_size = max(face_right - face_left, face_bottom - face_top)
                
                # If hand is close to face, likely phone usage
                if distance < face_size * 1.5:
                    confidence = 1.0 - (distance / (face_size * 1.5))
                    max_confidence = max(max_confidence, confidence)
                    
                    if confidence > PHONE_DETECTION_CONFIDENCE:
                        phone_detected = True
            
            return {
                'phone_detected': phone_detected,
                'confidence': max_confidence
            }
            
        except Exception as e:
            logging.error(f"Error detecting phone usage: {e}")
            return {'phone_detected': False, 'confidence': 0.0}
    
    def analyze_posture(self, frame):
        """Analyze body posture using MediaPipe Pose"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            pose_results = self.pose.process(rgb_frame)
            
            if not pose_results.pose_landmarks:
                return {'posture_score': 0.0, 'posture_issues': []}
            
            landmarks = pose_results.pose_landmarks.landmark
            
            # Analyze key posture points
            posture_issues = []
            posture_score = 1.0
            
            # Check shoulder alignment
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            if shoulder_diff > 0.1:  # Shoulders not level
                posture_issues.append("Uneven shoulders")
                posture_score -= 0.2
            
            # Check head position relative to shoulders
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            if nose.y > shoulder_center_y + 0.1:  # Head tilted down
                posture_issues.append("Head tilted down")
                posture_score -= 0.3
            
            # Check spine alignment
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            hip_diff = abs(left_hip.y - right_hip.y)
            if hip_diff > 0.1:  # Hips not level
                posture_issues.append("Uneven hips")
                posture_score -= 0.2
            
            posture_score = max(0.0, posture_score)
            
            return {
                'posture_score': posture_score,
                'posture_issues': posture_issues
            }
            
        except Exception as e:
            logging.error(f"Error analyzing posture: {e}")
            return {'posture_score': 0.0, 'posture_issues': []}
    
    def calculate_attention_score(self, face_landmarks, eye_ars, head_pose, sleeping_detection, phone_detection, posture_score):
        """Calculate overall attention score"""
        try:
            attention_score = 1.0
            
            # Eye aspect ratio contribution (30%)
            left_ear, right_ear = eye_ars
            avg_ear = (left_ear + right_ear) / 2
            if avg_ear < self.EYE_AR_THRESH:
                attention_score -= 0.3
            
            # Head pose contribution (25%)
            head_pose_penalty = 0.0
            if abs(head_pose['yaw']) > HEAD_POSE_THRESHOLD:
                head_pose_penalty += 0.15
            if abs(head_pose['roll']) > HEAD_POSE_THRESHOLD:
                head_pose_penalty += 0.1
            attention_score -= head_pose_penalty
            
            # Sleeping detection contribution (20%)
            if sleeping_detection['is_sleeping']:
                attention_score -= 0.2
            
            # Phone usage contribution (15%)
            if phone_detection['phone_detected']:
                attention_score -= 0.15
            
            # Posture contribution (10%)
            attention_score -= (1.0 - posture_score) * 0.1
            
            # Ensure score is between 0 and 1
            attention_score = max(0.0, min(1.0, attention_score))
            
            return attention_score
            
        except Exception as e:
            logging.error(f"Error calculating attention score: {e}")
            return 0.0
    
    def analyze_attention(self, frame, face_location):
        """Main function to analyze attention for a detected face"""
        try:
            # Extract face region
            top, right, bottom, left = face_location
            face_region = frame[top:bottom, left:right]
            
            if face_region.size == 0:
                return {
                    'attention_score': 0.0,
                    'distraction_score': 1.0,
                    'head_pose': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
                    'eye_aspect_ratio': 0.0,
                    'is_sleeping': False,
                    'phone_detected': False,
                    'posture_score': 0.0,
                    'face_detected': False
                }
            
            # Convert to RGB for MediaPipe
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks
            face_results = self.face_mesh.process(rgb_face)
            
            if not face_results.multi_face_landmarks:
                return {
                    'attention_score': 0.0,
                    'distraction_score': 1.0,
                    'head_pose': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
                    'eye_aspect_ratio': 0.0,
                    'is_sleeping': False,
                    'phone_detected': False,
                    'posture_score': 0.0,
                    'face_detected': False
                }
            
            # Get landmarks for the first face
            landmarks = face_results.multi_face_landmarks[0].landmark
            
            # Convert landmarks to pixel coordinates
            h, w = face_region.shape[:2]
            landmarks_px = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
            
            # Calculate eye aspect ratios
            left_eye_landmarks = landmarks_px[self.FACE_LANDMARKS['left_eye']]
            right_eye_landmarks = landmarks_px[self.FACE_LANDMARKS['right_eye']]
            
            left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks)
            right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks)
            
            # Calculate head pose
            head_pose = self.calculate_head_pose(landmarks_px)
            
            # Detect sleeping
            sleeping_detection = self.detect_sleeping(left_ear, right_ear, head_pose)
            
            # Detect phone usage
            phone_detection = self.detect_phone_usage(frame, face_location)
            
            # Analyze posture
            posture_analysis = self.analyze_posture(frame)
            
            # Calculate attention score
            attention_score = self.calculate_attention_score(
                landmarks_px, (left_ear, right_ear), head_pose, 
                sleeping_detection, phone_detection, posture_analysis['posture_score']
            )
            
            # Calculate distraction score
            distraction_score = 1.0 - attention_score
            
            return {
                'attention_score': attention_score,
                'distraction_score': distraction_score,
                'head_pose': head_pose,
                'eye_aspect_ratio': (left_ear + right_ear) / 2,
                'is_sleeping': sleeping_detection['is_sleeping'],
                'phone_detected': phone_detection['phone_detected'],
                'posture_score': posture_analysis['posture_score'],
                'face_detected': True,
                'sleeping_score': sleeping_detection['sleeping_score'],
                'phone_confidence': phone_detection['confidence']
            }
            
        except Exception as e:
            logging.error(f"Error analyzing attention: {e}")
            return {
                'attention_score': 0.0,
                'distraction_score': 1.0,
                'head_pose': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
                'eye_aspect_ratio': 0.0,
                'is_sleeping': False,
                'phone_detected': False,
                'posture_score': 0.0,
                'face_detected': False
            }
    
    def detect_cheating(self, attention_data, exam_mode=False):
        """Detect potential cheating behavior"""
        try:
            cheating_probability = 0.0
            cheating_indicators = []
            
            if not exam_mode:
                return {'cheating_probability': 0.0, 'indicators': []}
            
            # Check for phone usage
            if attention_data.get('phone_detected', False):
                cheating_probability += 0.4
                cheating_indicators.append("Phone usage detected")
            
            # Check for excessive head movement (looking around)
            head_pose = attention_data.get('head_pose', {})
            if abs(head_pose.get('yaw', 0)) > 45:
                cheating_probability += 0.3
                cheating_indicators.append("Excessive head movement")
            
            # Check for sleeping
            if attention_data.get('is_sleeping', False):
                cheating_probability += 0.2
                cheating_indicators.append("Sleeping detected")
            
            # Check for very low attention score
            if attention_data.get('attention_score', 1.0) < 0.3:
                cheating_probability += 0.3
                cheating_indicators.append("Very low attention")
            
            # Check for looking down frequently (might be looking at phone/notes)
            if abs(head_pose.get('pitch', 0)) > 30:
                cheating_probability += 0.2
                cheating_indicators.append("Looking down frequently")
            
            cheating_probability = min(1.0, cheating_probability)
            
            return {
                'cheating_probability': cheating_probability,
                'indicators': cheating_indicators
            }
            
        except Exception as e:
            logging.error(f"Error detecting cheating: {e}")
            return {'cheating_probability': 0.0, 'indicators': []}

