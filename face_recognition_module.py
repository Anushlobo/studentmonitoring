#face_recognition_module.py
import cv2
import numpy as np
import os
import pickle
import logging
import mediapipe as mp
from typing import List, Tuple, Dict, Optional
from config import DATASET_PATH, FACE_RECOGNITION_TOLERANCE, MIN_FACE_SIZE

class FaceRecognitionModule:
    def __init__(self):
        """Initialize the face recognition module using MediaPipe"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.model_path = "models/face_recognition_model.pkl"
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load or train the face recognition model
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train new one from dataset"""
        try:
            if os.path.exists(self.model_path):
                self.load_model()
                logging.info(f"Loaded face recognition model with {len(self.known_face_names)} known faces")
            else:
                self.train_model_from_dataset()
                logging.info(f"Trained face recognition model with {len(self.known_face_names)} known faces")
        except Exception as e:
            logging.error(f"Error in load_or_train_model: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
    
    def extract_face_features(self, image):
        """Extract face features using MediaPipe Face Mesh"""
        try:
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get face mesh landmarks
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            # Use the first face found
            landmarks = results.multi_face_landmarks[0]
            
            # Extract key facial landmarks for feature vector
            feature_vector = []
            
            # Key landmark indices for face recognition
            key_landmarks = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            
            for idx in key_landmarks:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    feature_vector.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(feature_vector)
            
        except Exception as e:
            logging.error(f"Error extracting face features: {e}")
            return None
    
    def train_model_from_dataset(self):
        """Train face recognition model from the dataset structure"""
        try:
            if not os.path.exists(DATASET_PATH):
                logging.warning(f"Dataset path {DATASET_PATH} does not exist")
                return
            
            # Iterate through student folders
            for student_folder in os.listdir(DATASET_PATH):
                student_path = os.path.join(DATASET_PATH, student_folder)
                
                if os.path.isdir(student_path):
                    student_id = student_folder
                    logging.info(f"Training model for student: {student_id}")
                    
                    # Get all images for this student
                    image_files = [f for f in os.listdir(student_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if not image_files:
                        logging.warning(f"No images found for student {student_id}")
                        continue
                    
                    # Process each image
                    for image_file in image_files:
                        image_path = os.path.join(student_path, image_file)
                        
                        try:
                            # Load and encode the image
                            image = cv2.imread(image_path)
                            if image is None:
                                logging.warning(f"Could not load image: {image_path}")
                                continue
                            
                            face_features = self.extract_face_features(image)
                            
                            if face_features is not None:
                                self.known_face_encodings.append(face_features)
                                self.known_face_names.append(student_id)
                                logging.info(f"Added face encoding for {student_id} from {image_file}")
                            else:
                                logging.warning(f"No face found in {image_path}")
                                
                        except Exception as e:
                            logging.error(f"Error processing {image_path}: {e}")
            
            # Save the trained model
            self.save_model()
            
        except Exception as e:
            logging.error(f"Error training model from dataset: {e}")
    
    def save_model(self):
        """Save the trained face recognition model"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'known_face_encodings': self.known_face_encodings,
                'known_face_names': self.known_face_names
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logging.info(f"Face recognition model saved to {self.model_path}")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load the trained face recognition model"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.known_face_encodings = model_data['known_face_encodings']
            self.known_face_names = model_data['known_face_names']
            
            logging.info(f"Face recognition model loaded from {self.model_path}")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
    
    def detect_faces(self, frame):
        """Detect faces in the frame using MediaPipe"""
        try:
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_results = self.face_detection.process(rgb_frame)
            
            face_locations = []
            face_encodings = []
            
            if face_results.detections:
                for detection in face_results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    face_location = (y, x + width, y + height, x)
                    face_locations.append(face_location)
                    
                    # Extract face region
                    face_region = frame[y:y+height, x:x+width]
                    if face_region.size > 0:
                        face_features = self.extract_face_features(face_region)
                        if face_features is not None:
                            face_encodings.append(face_features)
                        else:
                            face_encodings.append(None)
                    else:
                        face_encodings.append(None)
            
            return face_locations, face_encodings
            
        except Exception as e:
            logging.error(f"Error detecting faces: {e}")
            return [], []
    
    def compare_faces(self, face_encoding1, face_encoding2, tolerance=FACE_RECOGNITION_TOLERANCE):
        """Compare two face encodings using Euclidean distance"""
        try:
            if face_encoding1 is None or face_encoding2 is None:
                return False
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(face_encoding1 - face_encoding2)
            
            # Return True if distance is less than tolerance
            return distance < tolerance
            
        except Exception as e:
            logging.error(f"Error comparing faces: {e}")
            return False
    
    def recognize_students(self, frame):
        """Recognize students in the frame and return results"""
        try:
            face_locations, face_encodings = self.detect_faces(frame)
            
            # Create results dictionary
            recognition_results = []
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = face_location
                
                # Calculate confidence based on face size
                face_width = right - left
                face_height = bottom - top
                face_size = face_width * face_height
                
                confidence = min(1.0, face_size / (MIN_FACE_SIZE * MIN_FACE_SIZE))
                
                # Find matching known face
                name = "Unknown"
                best_match_distance = float('inf')
                
                if face_encoding is not None:
                    # Import tolerance dynamically to get updated value
                    from config import FACE_RECOGNITION_TOLERANCE
                    for known_encoding, known_name in zip(self.known_face_encodings, self.known_face_names):
                        distance = np.linalg.norm(face_encoding - known_encoding)
                        if distance < best_match_distance and distance < FACE_RECOGNITION_TOLERANCE:
                            best_match_distance = distance
                            name = known_name
                            confidence = max(confidence, 1.0 - distance / FACE_RECOGNITION_TOLERANCE)
                
                result = {
                    'student_id': name,
                    'confidence': confidence,
                    'face_location': face_location,
                    'face_detected': True,
                    'face_index': i
                }
                
                recognition_results.append(result)
            
            return recognition_results
            
        except Exception as e:
            logging.error(f"Error recognizing students: {e}")
            return []
    
    def add_student_face(self, student_id: str, face_image: np.ndarray) -> bool:
        """Add a new face for a student"""
        try:
            # Extract face features
            face_features = self.extract_face_features(face_image)
            
            if face_features is None:
                logging.warning(f"No face found in the provided image for student {student_id}")
                return False
            
            # Add to known faces
            self.known_face_encodings.append(face_features)
            self.known_face_names.append(student_id)
            
            # Save updated model
            self.save_model()
            
            logging.info(f"Added new face for student {student_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding student face: {e}")
            return False
    
    def get_known_students(self) -> List[str]:
        """Get list of known student IDs"""
        return list(set(self.known_face_names))
    
    def remove_student(self, student_id: str) -> bool:
        """Remove a student from the recognition system"""
        try:
            # Find all indices for this student
            indices_to_remove = [i for i, name in enumerate(self.known_face_names) if name == student_id]
            
            # Remove in reverse order to maintain indices
            for index in reversed(indices_to_remove):
                del self.known_face_names[index]
                del self.known_face_encodings[index]
            
            # Save updated model
            self.save_model()
            
            logging.info(f"Removed student {student_id} from recognition system")
            return True
            
        except Exception as e:
            logging.error(f"Error removing student: {e}")
            return False
    
    def get_recognition_stats(self) -> Dict:
        """Get statistics about the recognition system"""
        try:
            unique_students = set(self.known_face_names)
            student_face_counts = {}
            
            for student_id in unique_students:
                student_face_counts[student_id] = self.known_face_names.count(student_id)
            
            return {
                'total_faces': len(self.known_face_encodings),
                'unique_students': len(unique_students),
                'student_face_counts': student_face_counts
            }
            
        except Exception as e:
            logging.error(f"Error getting recognition stats: {e}")
            return {}
