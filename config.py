import os

# System Configuration
SYSTEM_NAME = "AI-Based Student Monitoring System"
VERSION = "1.0.0"

# Video Processing Configuration
FRAME_INTERVAL = 30  # Process 1 frame every 30 seconds
VIDEO_SOURCE = 0  # Default camera (0 for webcam, or path to video file)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FPS = 30

# Face Recognition Configuration
FACE_RECOGNITION_TOLERANCE = 2.0
MIN_FACE_SIZE = 20
FACE_DETECTION_CONFIDENCE = 0.5

# Attention Analysis Configuration
ATTENTION_THRESHOLD = 0.7
DISTRACTION_THRESHOLD = 0.3
EYE_ASPECT_RATIO_THRESHOLD = 0.2
HEAD_POSE_THRESHOLD = 30  # degrees

# YOLO Configuration
YOLO_MODEL_PATH = "models/yolov8n.pt"
PERSON_CONFIDENCE_THRESHOLD = 0.5

# Database Configuration
DATABASE_PATH = "data/monitoring.db"
CSV_LOG_PATH = "data/monitoring_logs.csv"

# Dataset Configuration
DATASET_PATH = "dataset"
STUDENT_IMAGES_PER_PERSON = 10  # Minimum images per student for training

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/monitoring.log"

# Analysis Windows
ATTENTION_WINDOW_MINUTES = 5  # Analyze attention over 5-minute windows
ATTENDANCE_WINDOW_MINUTES = 2  # Consider attendance valid for 2 minutes

# Cheating Detection
CHEATING_DETECTION_ENABLED = True
PHONE_DETECTION_CONFIDENCE = 0.7
SLEEPING_DETECTION_THRESHOLD = 0.8

# Output Configuration
SAVE_ANALYSIS_FRAMES = True
ANALYSIS_FRAMES_PATH = "output/analysis_frames"
SAVE_ATTENDANCE_REPORTS = True
ATTENDANCE_REPORTS_PATH = "output/attendance_reports"

# Create necessary directories
def create_directories():
    """Create necessary directories for the system"""
    directories = [
        "dataset",
        "data",
        "logs",
        "models",
        "output",
        "output/analysis_frames",
        "output/attendance_reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories
create_directories()

