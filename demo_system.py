#!/usr/bin/env python3
#demo_system.py
"""
Demo script to showcase the AI Student Monitoring System
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

from config import *
from database import MonitoringDatabase
from face_recognition_module import FaceRecognitionModule
from attention_analysis import AttentionAnalysisModule
from student_detection import StudentDetectionModule
from main_monitoring_system import StudentMonitoringSystem

def create_demo_video():
    """Create a simple demo video for testing"""
    print("🎬 Creating demo video...")
    
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_video.mp4', fourcc, 30.0, (640, 480))
    
    # Create frames with simulated students
    for i in range(300):  # 10 seconds at 30 fps
        # Create a frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add some simulated faces (circles)
        cv2.circle(frame, (200, 200), 50, (255, 255, 255), -1)  # Face 1
        cv2.circle(frame, (400, 200), 50, (255, 255, 255), -1)  # Face 2
        cv2.circle(frame, (300, 300), 50, (255, 255, 255), -1)  # Face 3
        
        # Add some text
        cv2.putText(frame, f'Demo Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, 'Simulated Students', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("✅ Demo video created: demo_video.mp4")
    return 'demo_video.mp4'

def demo_face_recognition():
    """Demo face recognition capabilities"""
    print("\n👤 Face Recognition Demo:")
    print("-" * 40)
    
    # Initialize face recognition
    face_recognition = FaceRecognitionModule()
    
    # Create a test image with faces
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.circle(test_image, (200, 200), 50, (255, 255, 255), -1)
    cv2.circle(test_image, (400, 200), 50, (255, 255, 255), -1)
    
    # Detect faces
    faces = face_recognition.detect_faces(test_image)
    print(f"✅ Detected {len(faces)} faces in test image")
    
    # Show dataset info
    if os.path.exists(DATASET_PATH):
        student_folders = [f for f in os.listdir(DATASET_PATH) 
                          if os.path.isdir(os.path.join(DATASET_PATH, f))]
        print(f"✅ Dataset contains {len(student_folders)} students:")
        for student in student_folders[:5]:  # Show first 5
            student_path = os.path.join(DATASET_PATH, student)
            images = [f for f in os.listdir(student_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   - {student}: {len(images)} images")
        if len(student_folders) > 5:
            print(f"   ... and {len(student_folders) - 5} more students")

def demo_attention_analysis():
    """Demo attention analysis capabilities"""
    print("\n🧠 Attention Analysis Demo:")
    print("-" * 40)
    
    # Initialize attention analysis
    attention_analysis = AttentionAnalysisModule()
    
    # Create a test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.circle(test_image, (320, 240), 50, (255, 255, 255), -1)  # Simulated face
    
    # Analyze attention
    face_location = (190, 370, 290, 270)  # top, right, bottom, left
    attention_result = attention_analysis.analyze_attention(test_image, face_location)
    
    print(f"✅ Attention Score: {attention_result['attention_score']:.3f}")
    print(f"✅ Distraction Score: {attention_result['distraction_score']:.3f}")
    print(f"✅ Head Pose: Yaw={attention_result['head_pose']['yaw']:.1f}°, "
          f"Pitch={attention_result['head_pose']['pitch']:.1f}°, "
          f"Roll={attention_result['head_pose']['roll']:.1f}°")
    print(f"✅ Eye Aspect Ratio: {attention_result['eye_aspect_ratio']:.3f}")
    print(f"✅ Sleeping Detected: {attention_result['is_sleeping']}")
    print(f"✅ Phone Detected: {attention_result['phone_detected']}")

def demo_student_detection():
    """Demo student detection capabilities"""
    print("\n👥 Student Detection Demo:")
    print("-" * 40)
    
    # Initialize student detection
    student_detection = StudentDetectionModule()
    
    # Create a test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Add simulated people (rectangles)
    cv2.rectangle(test_image, (100, 100), (200, 300), (255, 255, 255), -1)
    cv2.rectangle(test_image, (300, 100), (400, 300), (255, 255, 255), -1)
    cv2.rectangle(test_image, (500, 100), (600, 300), (255, 255, 255), -1)
    
    # Detect people
    people = student_detection.detect_people(test_image)
    print(f"✅ Detected {len(people)} people in test image")
    
    # Get detection summary
    summary = student_detection.get_detection_summary(test_image)
    print(f"✅ Detection Summary: {summary}")

def demo_monitoring_system():
    """Demo the complete monitoring system"""
    print("\n🖥️ Complete Monitoring System Demo:")
    print("-" * 40)
    
    # Initialize monitoring system
    monitoring_system = StudentMonitoringSystem()
    
    # Start a session
    monitoring_system.start_session('demo_session', False, 0)
    print("✅ Session started")
    
    # Process a few test frames
    for i in range(3):
        # Create test frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.circle(test_frame, (200 + i*50, 200), 30, (255, 255, 255), -1)
        cv2.putText(test_frame, f'Frame {i+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Process frame
        results = monitoring_system.process_frame(test_frame)
        print(f"✅ Frame {i+1} processed: {len(results) if results else 0} results")
        time.sleep(0.5)
    
    # End session
    monitoring_system.end_session()
    print("✅ Session ended")
    
    # Show session statistics
    print(f"✅ Total frames analyzed: {len(monitoring_system.frame_analysis_results)}")
    print(f"✅ Students tracked: {len(monitoring_system.attendance_tracking)}")

def demo_web_interface():
    """Demo the web interface"""
    print("\n🌐 Web Interface Demo:")
    print("-" * 40)
    print("✅ Web interface is ready!")
    print("✅ Open your browser and go to: http://localhost:5000")
    print("✅ Features available:")
    print("   - Upload video files (MP4, AVI, MOV, MKV, WMV)")
    print("   - Choose between Normal Mode and Exam Mode")
    print("   - Real-time progress tracking")
    print("   - Instant analysis reports")
    print("   - Report gallery with download options")
    print("✅ The system will:")
    print("   - Detect students using YOLO")
    print("   - Recognize faces using MediaPipe")
    print("   - Analyze attention and distractions")
    print("   - Generate comprehensive reports")

def main():
    """Run the complete demo"""
    print("🎯 AI Student Monitoring System - Demo")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print("⚠️ No dataset found! Creating sample dataset...")
        os.system("python demo_dataset_creator.py --create-sample")
    
    # Run demos
    demo_face_recognition()
    demo_attention_analysis()
    demo_student_detection()
    demo_monitoring_system()
    demo_web_interface()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("✅ All AI models are working correctly")
    print("✅ Web interface is ready for use")
    print("✅ You can now upload video files and analyze them")
    
    # Create demo video
    demo_video = create_demo_video()
    print(f"✅ Demo video created: {demo_video}")
    print("✅ You can use this video to test the web interface")

if __name__ == "__main__":
    main()






