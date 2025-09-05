#!/usr/bin/env python3
#run_web.py
"""
Web Application Runner for AI Student Monitoring System
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import cv2
        import mediapipe
        import ultralytics
        import numpy
        import pandas
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'templates',
        'uploads',
        'reports',
        'static',
        'static/reports',
        'dataset',
        'data',
        'logs',
        'models',
        'output',
        'output/analysis_frames',
        'output/attendance_reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    """Main function to run the web application"""
    print("🚀 Starting AI Student Monitoring System Web Application")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first:")
        print("   python setup_environment.py")
        return 1
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    # Check if dataset exists
    if not os.path.exists('dataset') or not os.listdir('dataset'):
        print("\n⚠️  Warning: No dataset found!")
        print("   Please create a dataset with student images:")
        print("   python demo_dataset_creator.py --create-sample")
        print("   Or add your own student images to the dataset/ directory")
    
    # Start the web application
    print("\n🌐 Starting web server...")
    print("   The application will be available at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

