#!/usr/bin/env python3
"""
Demo Dataset Creator
Utility script to create sample datasets for testing the AI-Based Student Monitoring System
"""

import os
import cv2
import numpy as np
from datetime import datetime
import argparse
import logging

def create_sample_dataset():
    """Create a sample dataset structure with placeholder images"""
    
    # Create dataset directory
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Sample student names
    sample_students = [
        "student_001",
        "student_002", 
        "student_003",
        "student_004",
        "student_005"
    ]
    
    print("🎯 Creating sample dataset...")
    print(f"📁 Dataset directory: {dataset_dir}")
    
    for i, student_name in enumerate(sample_students):
        student_dir = os.path.join(dataset_dir, student_name)
        os.makedirs(student_dir, exist_ok=True)
        
        print(f"👤 Creating folder for {student_name}...")
        
        # Create 5 sample images for each student
        for j in range(5):
            # Create a simple placeholder image
            img = create_placeholder_image(f"{student_name} - Image {j+1}")
            
            # Save image
            img_path = os.path.join(student_dir, f"img{j+1}.jpg")
            cv2.imwrite(img_path, img)
            
            print(f"  📸 Created {img_path}")
    
    print(f"\n✅ Sample dataset created successfully!")
    print(f"📊 Total students: {len(sample_students)}")
    print(f"📸 Total images: {len(sample_students) * 5}")
    print(f"\n📝 Next steps:")
    print(f"1. Replace placeholder images with actual student photos")
    print(f"2. Ensure each student has 5-10 clear face images")
    print(f"3. Run the monitoring system to test face recognition")

def create_placeholder_image(text):
    """Create a placeholder image with text"""
    # Create a 640x480 image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 0)  # Black text
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Center the text
    x = (img.shape[1] - text_width) // 2
    y = (img.shape[0] + text_height) // 2
    
    # Add text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
    
    # Add border
    cv2.rectangle(img, (10, 10), (img.shape[1]-10, img.shape[0]-10), (100, 100, 100), 3)
    
    return img

def create_camera_capture_dataset():
    """Create dataset by capturing images from camera"""
    
    print("📷 Camera Capture Mode")
    print("This will help you capture real student images for the dataset.")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Create dataset directory
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\n📝 Instructions:")
    print("1. Enter student name/ID when prompted")
    print("2. Position face in camera view")
    print("3. Press 'c' to capture image")
    print("4. Press 'q' to quit")
    print("5. Press 'n' for next student")
    
    current_student = None
    image_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading camera")
            break
        
        # Display frame
        cv2.imshow('Camera Capture - Press c to capture, q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            if current_student:
                # Capture image
                image_count += 1
                img_path = os.path.join(dataset_dir, current_student, f"img{image_count}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"📸 Captured: {img_path}")
            else:
                print("❌ Please enter student name first (press 'n')")
        elif key == ord('n'):
            # New student
            student_name = input("\n👤 Enter student name/ID: ").strip()
            if student_name:
                current_student = student_name
                student_dir = os.path.join(dataset_dir, student_name)
                os.makedirs(student_dir, exist_ok=True)
                image_count = 0
                print(f"✅ Ready to capture images for {student_name}")
            else:
                print("❌ Invalid student name")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Camera capture completed!")

def validate_dataset():
    """Validate existing dataset structure"""
    
    dataset_dir = "dataset"
    
    if not os.path.exists(dataset_dir):
        print("❌ Dataset directory not found")
        return False
    
    print("🔍 Validating dataset...")
    
    student_folders = [f for f in os.listdir(dataset_dir) 
                      if os.path.isdir(os.path.join(dataset_dir, f))]
    
    if not student_folders:
        print("❌ No student folders found in dataset")
        return False
    
    total_images = 0
    valid_students = 0
    
    for student_folder in student_folders:
        student_path = os.path.join(dataset_dir, student_folder)
        image_files = [f for f in os.listdir(student_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) >= 3:  # Minimum 3 images per student
            valid_students += 1
            total_images += len(image_files)
            print(f"✅ {student_folder}: {len(image_files)} images")
        else:
            print(f"⚠️  {student_folder}: {len(image_files)} images (minimum 3 recommended)")
    
    print(f"\n📊 Dataset Summary:")
    print(f"Total students: {len(student_folders)}")
    print(f"Valid students: {valid_students}")
    print(f"Total images: {total_images}")
    
    if valid_students > 0:
        print(f"✅ Dataset is ready for use!")
        return True
    else:
        print(f"❌ Dataset needs more images")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Demo Dataset Creator for Student Monitoring System')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset with placeholder images')
    parser.add_argument('--camera-capture', action='store_true', help='Create dataset by capturing from camera')
    parser.add_argument('--validate', action='store_true', help='Validate existing dataset')
    
    args = parser.parse_args()
    
    print("🎯 Demo Dataset Creator")
    print("=" * 50)
    
    if args.create_sample:
        create_sample_dataset()
    elif args.camera_capture:
        create_camera_capture_dataset()
    elif args.validate:
        validate_dataset()
    else:
        # Interactive mode
        print("📋 Available options:")
        print("1. Create sample dataset (placeholder images)")
        print("2. Camera capture mode (real images)")
        print("3. Validate existing dataset")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            create_sample_dataset()
        elif choice == '2':
            create_camera_capture_dataset()
        elif choice == '3':
            validate_dataset()
        elif choice == '4':
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()

