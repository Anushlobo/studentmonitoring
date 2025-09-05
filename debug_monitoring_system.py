#!/usr/bin/env python3
"""
Debug script to identify where the monitoring system is failing
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime

def debug_yolo():
    """Debug YOLO detection specifically"""
    print("🔍 Debugging YOLO Detection...")
    
    try:
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully")
        
        # Use a real image from the dataset instead of synthetic shapes
        test_image_path = 'dataset/anush/anush.jpg'
        test_image = cv2.imread(test_image_path)
        
        if test_image is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return False
        
        print(f"✅ Using real test image: {test_image_path}, shape: {test_image.shape}")
        
        # Save test image for inspection
        cv2.imwrite('debug_test_image.jpg', test_image)
        print("✅ Test image saved as debug_test_image.jpg")
        
        # Run YOLO detection
        results = model(test_image, verbose=True)
        
        print(f"✅ YOLO Results: {len(results)} result objects")
        
        detections = []
        for result in results:
            print(f"   - Result type: {type(result)}")
            boxes = result.boxes
            print(f"   - Boxes: {boxes}")
            
            if boxes is not None:
                print(f"   - Number of boxes: {len(boxes)}")
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    print(f"   - Box {i}: class_id={class_id}, conf={confidence:.3f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                    
                    if class_id == 0:  # Person class
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(confidence),
                            'class_id': class_id
                        })
            else:
                print("   - No boxes detected")
        
        print(f"✅ Final detections: {len(detections)} people found")
        for i, det in enumerate(detections):
            print(f"   Person {i+1}: {det}")
        
        return len(detections) > 0
        
    except Exception as e:
        print(f"❌ YOLO Debug Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_student_detection_module():
    """Debug the StudentDetectionModule"""
    print("\n👥 Debugging Student Detection Module...")
    
    try:
        from student_detection import StudentDetectionModule
        
        detector = StudentDetectionModule()
        
        if detector.model is None:
            print("❌ YOLO model failed to load in StudentDetectionModule")
            return False
        
        print("✅ StudentDetectionModule initialized")
        
        # Use a real image from the dataset
        test_image_path = 'dataset/anush/anush.jpg'
        test_image = cv2.imread(test_image_path)
        
        if test_image is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return False
        
        print(f"✅ Using real test image: {test_image_path}, shape: {test_image.shape}")
        
        # Test detection
        people = detector.detect_people(test_image)
        print(f"✅ detect_people returned: {len(people)} people")
        for i, person in enumerate(people):
            print(f"   Person {i+1}: {person}")
        
        # Test filtering
        students = detector.filter_students(people)
        print(f"✅ filter_students returned: {len(students)} students")
        for i, student in enumerate(students):
            print(f"   Student {i+1}: {student}")
        
        return len(students) > 0
        
    except Exception as e:
        print(f"❌ Student Detection Module Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_face_recognition():
    """Debug face recognition"""
    print("\n👤 Debugging Face Recognition...")
    
    try:
        from face_recognition_module import FaceRecognitionModule
        
        face_rec = FaceRecognitionModule()
        print("✅ FaceRecognitionModule initialized")
        
        # Use a real image from the dataset
        test_image_path = 'dataset/anush/anush.jpg'
        test_image = cv2.imread(test_image_path)
        
        if test_image is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return False
        
        print(f"✅ Using real test image: {test_image_path}, shape: {test_image.shape}")
        
        cv2.imwrite('debug_face_image.jpg', test_image)
        print("✅ Face test image saved as debug_face_image.jpg")
        
        # Test face detection
        face_locations, face_encodings = face_rec.detect_faces(test_image)
        print(f"✅ Face detection: {len(face_locations)} faces, {len(face_encodings)} encodings")
        
        # Test recognition
        recognition_results = face_rec.recognize_students(test_image)
        print(f"✅ Recognition results: {len(recognition_results)} results")
        for i, result in enumerate(recognition_results):
            print(f"   Result {i+1}: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Face Recognition Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_attention_analysis():
    """Debug attention analysis"""
    print("\n🧠 Debugging Attention Analysis...")
    
    try:
        from attention_analysis import AttentionAnalysisModule
        
        attention = AttentionAnalysisModule()
        print("✅ AttentionAnalysisModule initialized")
        
        # Use a real image from the dataset
        test_image_path = 'dataset/anush/anush.jpg'
        test_image = cv2.imread(test_image_path)
        
        if test_image is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return False
        
        print(f"✅ Using real test image: {test_image_path}, shape: {test_image.shape}")
        
        # Test with a face location (approximate face region from the image)
        face_location = (200, 600, 800, 200)  # top, right, bottom, left
        
        attention_result = attention.analyze_attention(test_image, face_location)
        print(f"✅ Attention analysis result: {attention_result}")
        
        return attention_result['face_detected']
        
    except Exception as e:
        print(f"❌ Attention Analysis Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_frame_processing():
    """Debug the main frame processing logic"""
    print("\n🖥️ Debugging Frame Processing...")
    
    try:
        from main_monitoring_system import StudentMonitoringSystem
        
        system = StudentMonitoringSystem()
        print("✅ StudentMonitoringSystem initialized")
        
        # Start a session
        session_id = system.start_session('debug_session', False, 0)
        print(f"✅ Session started: {session_id}")
        
        # Use a real image from the dataset
        test_image_path = 'dataset/anush/anush.jpg'
        test_frame = cv2.imread(test_image_path)
        
        if test_frame is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return False
        
        print(f"✅ Using real test image: {test_image_path}, shape: {test_frame.shape}")
        
        cv2.imwrite('debug_frame.jpg', test_frame)
        print("✅ Debug frame saved as debug_frame.jpg")
        
        # Process the frame (bypass frame interval check for testing)
        print("🔄 Processing frame...")
        
        # Manually set frame count to bypass interval check
        system.frame_count = 900  # Set to a high number to bypass interval check
        system.last_processed_frame = 0  # Reset last processed frame
        
        results = system.process_frame(test_frame)
        
        print(f"✅ Frame processing results: {len(results) if results else 0} results")
        if results:
            for i, result in enumerate(results):
                print(f"   Result {i+1}: student_id={result.get('student_id')}, "
                      f"attention={result.get('attention_score', 0):.3f}")
        
        # Check if data was stored
        print(f"✅ Frame analysis results stored: {len(system.frame_analysis_results)}")
        print(f"✅ Attendance tracking: {len(system.attendance_tracking)} students")
        
        # End session
        system.end_session()
        print("✅ Session ended")
        
        return len(results) > 0 if results else False
        
    except Exception as e:
        print(f"❌ Frame Processing Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_database():
    """Debug database operations"""
    print("\n🗃️ Debugging Database...")
    
    try:
        from database import MonitoringDatabase
        
        db = MonitoringDatabase()
        print("✅ Database initialized")
        
        # Test logging some data
        test_data = {
            'student_id': 'debug_student',
            'face_detected': True,
            'attention_score': 0.8,
            'distraction_score': 0.2,
            'head_pose_x': 0.0,
            'head_pose_y': 5.0,
            'head_pose_z': 0.0,
            'eye_aspect_ratio': 0.3,
            'is_sleeping': False,
            'phone_detected': False,
            'cheating_probability': 0.1,
            'frame_number': 1,
            'session_id': 'debug_session'
        }
        
        success = db.log_monitoring_data(test_data)
        print(f"✅ Database logging test: {'SUCCESS' if success else 'FAILED'}")
        
        # Check if CSV was created
        if os.path.exists(db.csv_path):
            with open(db.csv_path, 'r') as f:
                lines = f.readlines()
                print(f"✅ CSV file has {len(lines)} lines")
        else:
            print("❌ CSV file not created")
        
        return success
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_video_processing():
    """Debug video processing from app.py"""
    print("\n🎬 Debugging Video Processing...")
    
    try:
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('debug_test_video.mp4', fourcc, 5.0, (640, 480))
        
        # Create 25 frames (5 seconds at 5 fps)
        for i in range(25):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
            
            # Add moving person
            x_pos = 100 + (i * 20)  # Moving person
            cv2.rectangle(frame, (x_pos, 100), (x_pos + 80, 350), (200, 200, 200), -1)
            cv2.rectangle(frame, (x_pos + 15, 80), (x_pos + 65, 120), (200, 200, 200), -1)
            
            # Add frame number
            cv2.putText(frame, f'Frame {i+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print("✅ Debug video created: debug_test_video.mp4")
        
        # Test opening the video
        cap = cv2.VideoCapture('debug_test_video.mp4')
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"✅ Video opened: {total_frames} frames, {fps} fps")
            
            # Test reading a few frames
            frames_read = 0
            while frames_read < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_read += 1
                print(f"   Frame {frames_read}: {frame.shape}")
            
            cap.release()
            return True
        else:
            print("❌ Could not open created video")
            return False
        
    except Exception as e:
        print(f"❌ Video Processing Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🐛 COMPREHENSIVE DEBUG ANALYSIS")
    print("=" * 60)
    
    tests = [
        ("YOLO Detection", debug_yolo),
        ("Student Detection Module", debug_student_detection_module),
        ("Face Recognition", debug_face_recognition),
        ("Attention Analysis", debug_attention_analysis),
        ("Frame Processing", debug_frame_processing),
        ("Database Operations", debug_database),
        ("Video Processing", debug_video_processing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DEBUG SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ WORKING" if result else "❌ FAILING"
        print(f"{test_name}: {status}")
    
    # Identify the failure point
    print(f"\n🔍 FAILURE ANALYSIS:")
    failed_tests = [name for name, result in results.items() if not result]
    
    if not failed_tests:
        print("✅ All components are working! The issue might be in integration or configuration.")
    else:
        print(f"❌ Failed components: {', '.join(failed_tests)}")
        
        if "YOLO Detection" in failed_tests:
            print("🔧 Fix: Check YOLO model loading and installation")
        if "Student Detection Module" in failed_tests:
            print("🔧 Fix: Check StudentDetectionModule initialization")
        if "Frame Processing" in failed_tests:
            print("🔧 Fix: Check main processing pipeline logic")
        if "Database Operations" in failed_tests:
            print("🔧 Fix: Check database permissions and paths")

if __name__ == "__main__":
    main()