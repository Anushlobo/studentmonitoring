#!/usr/bin/env python3
#test_models.py
"""
Test script to verify AI models are working correctly
"""

import cv2
import numpy as np
import os
from datetime import datetime

# Import our modules
from config import *
from database import MonitoringDatabase
from face_recognition_module import FaceRecognitionModule
from attention_analysis import AttentionAnalysisModule
from student_detection import StudentDetectionModule
from main_monitoring_system import StudentMonitoringSystem

def test_mediapipe():
    """Test MediaPipe face detection and mesh"""
    print("🔍 Testing MediaPipe...")
    
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create a test image (simple colored rectangle)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Test face detection
        results_detection = mp_face_detection.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        print(f"   ✅ Face Detection: {len(results_detection.detections) if results_detection.detections else 0} faces detected")
        
        # Test face mesh
        results_mesh = mp_face_mesh.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        print(f"   ✅ Face Mesh: {len(results_mesh.multi_face_landmarks) if results_mesh.multi_face_landmarks else 0} faces detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ MediaPipe Error: {e}")
        return False

def test_yolo():
    """Test YOLO model loading"""
    print("🎯 Testing YOLO...")
    
    try:
        from ultralytics import YOLO
        
        # Load YOLO model
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("   ✅ YOLO model loaded successfully")
        
        # Test inference on a simple image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        results = model(test_image)
        print(f"   ✅ YOLO inference: {len(results[0].boxes) if results[0].boxes is not None else 0} objects detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ YOLO Error: {e}")
        return False

def test_face_recognition():
    """Test face recognition module"""
    print("👤 Testing Face Recognition...")
    
    try:
        # Initialize face recognition
        face_recognition = FaceRecognitionModule()
        print("   ✅ Face Recognition module initialized")
        
        # Test with a simple image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Test face detection
        faces = face_recognition.detect_faces(test_image)
        print(f"   ✅ Face Detection: {len(faces)} faces detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Face Recognition Error: {e}")
        return False

def test_attention_analysis():
    """Test attention analysis module"""
    print("🧠 Testing Attention Analysis...")
    
    try:
        # Initialize attention analysis
        attention_analysis = AttentionAnalysisModule()
        print("   ✅ Attention Analysis module initialized")
        
        # Test with a simple image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Test attention analysis with a face location (simulated)
        face_location = (100, 200, 300, 400)  # top, right, bottom, left
        attention_result = attention_analysis.analyze_attention(test_image, face_location)
        print(f"   ✅ Attention Analysis: Score = {attention_result['attention_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Attention Analysis Error: {e}")
        return False

def test_student_detection():
    """Test student detection module"""
    print("👥 Testing Student Detection...")
    
    try:
        # Initialize student detection
        student_detection = StudentDetectionModule()
        print("   ✅ Student Detection module initialized")
        
        # Test with a simple image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Test people detection
        people = student_detection.detect_people(test_image)
        print(f"   ✅ People Detection: {len(people)} people detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Student Detection Error: {e}")
        return False

def test_monitoring_system():
    """Test the main monitoring system"""
    print("🖥️ Testing Monitoring System...")
    
    try:
        # Initialize monitoring system
        monitoring_system = StudentMonitoringSystem()
        print("   ✅ Monitoring System initialized")
        
        # Start a session
        monitoring_system.start_session('test_session', False, 0)
        print("   ✅ Session started")
        
        # Test with a simple frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        results = monitoring_system.process_frame(test_frame)
        print(f"   ✅ Frame processed: {len(results) if results else 0} results")
        
        # End session
        monitoring_system.end_session()
        print("   ✅ Session ended")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Monitoring System Error: {e}")
        return False

def test_dataset():
    """Test if dataset exists and is accessible"""
    print("📁 Testing Dataset...")
    
    try:
        if os.path.exists(DATASET_PATH):
            student_folders = [f for f in os.listdir(DATASET_PATH) 
                            if os.path.isdir(os.path.join(DATASET_PATH, f))]
            print(f"   ✅ Dataset found: {len(student_folders)} student folders")
            
            if student_folders:
                # Check first student folder
                first_student = student_folders[0]
                student_path = os.path.join(DATASET_PATH, first_student)
                images = [f for f in os.listdir(student_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"   ✅ Sample student '{first_student}': {len(images)} images")
            else:
                print("   ⚠️ No student folders found in dataset")
        else:
            print("   ❌ Dataset directory not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 AI Model Testing Suite")
    print("=" * 50)
    
    tests = [
        ("MediaPipe", test_mediapipe),
        ("YOLO", test_yolo),
        ("Face Recognition", test_face_recognition),
        ("Attention Analysis", test_attention_analysis),
        ("Student Detection", test_student_detection),
        ("Monitoring System", test_monitoring_system),
        ("Dataset", test_dataset),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system should work correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
