# AI-Based Student Monitoring System

An advanced AI-powered system for monitoring student behavior, attention, and attendance in classroom environments. The system can detect multiple students (50+), track attention levels, identify distractions, and monitor for cheating during exams.

## 🚀 Features

### Core Capabilities
- **Multi-Student Detection**: Detect and track 50+ students simultaneously using YOLO
- **Face Recognition**: Automatic student identification using trained face recognition models
- **Attention Analysis**: Monitor posture, eye tracking, and head pose for attention assessment
- **Distraction Detection**: Identify phone usage, sleeping, and other distractions
- **Cheating Detection**: Enhanced monitoring during exam mode
- **Automatic Attendance**: Generate attendance reports using face recognition
- **Frame Rate Optimization**: Process 1 frame every 30 seconds for efficient analysis

### Technical Features
- **Real-time Processing**: Live video analysis with configurable frame intervals
- **Database Storage**: SQLite database for persistent data storage
- **CSV Logging**: Detailed logs for analysis and reporting
- **Session Management**: Support for different session types (regular, exam, lecture, workshop)
- **Report Generation**: Automatic generation of attendance and analysis reports

## 📁 Project Structure

```
newstudentmonitoringsystem/
├── config.py                      # System configuration
├── database.py                    # Database management
├── face_recognition_module.py     # Face recognition system
├── attention_analysis.py          # Attention and distraction analysis
├── student_detection.py           # YOLO-based student detection
├── main_monitoring_system.py     # Main monitoring orchestrator
├── run_monitoring.py             # Command-line interface
├── setup_environment.py          # Environment setup script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── dataset/                      # Student face dataset
│   ├── student_1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── student_2/
│   │   ├── img1.jpg
│   │   └── ...
│   └── ...
├── data/                         # Database and logs
├── logs/                         # System logs
├── models/                       # AI models
├── output/                       # Generated reports and analysis frames
│   ├── analysis_frames/
│   └── attendance_reports/
└── venv/                         # Virtual environment (created during setup)
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam or video input device
- Sufficient RAM (8GB+ recommended)
- GPU support (optional, for faster processing)

### Step 1: Clone and Setup Environment
```bash
# Navigate to project directory
cd newstudentmonitoringsystem

# Run the setup script
python setup_environment.py
```

### Step 2: Activate Virtual Environment
```bash
# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Prepare Student Dataset
Create the dataset structure as follows:

```
dataset/
├── student_1/          # Student name or ID
│   ├── img1.jpg        # Multiple images of the student
│   ├── img2.jpg        # Different angles and expressions
│   └── img3.jpg
├── student_2/
│   ├── img1.jpg
│   └── img2.jpg
└── ...
```

**Requirements for student images:**
- Clear, well-lit photos
- Multiple angles (front, side, different expressions)
- Minimum 5-10 images per student
- JPG/PNG format
- Face should be clearly visible

### Step 4: Configure System
Edit `config.py` to customize system settings:

```python
# Video source (0 for webcam, or path to video file)
VIDEO_SOURCE = 0

# Frame processing interval (seconds)
FRAME_INTERVAL = 30

# Face recognition tolerance
FACE_RECOGNITION_TOLERANCE = 0.6

# Attention thresholds
ATTENTION_THRESHOLD = 0.7
DISTRACTION_THRESHOLD = 0.3
```

## 🎯 Usage

### Web Application (Recommended)
```bash
# Start the web application
python run_web.py
```

Then open your browser and navigate to: **http://localhost:5000**

The web interface provides:
- **Two Analysis Modes**: Normal Mode and Exam Mode
- **Video Upload**: Drag & drop or browse for video files
- **Real-time Progress**: Live progress tracking during analysis
- **Instant Reports**: View results immediately after processing
- **Report Gallery**: Browse all generated reports
- **Download Reports**: Download detailed JSON reports

### Video File Requirements
- **Supported Formats**: MP4, AVI, MOV, MKV, WMV
- **Maximum Size**: 500MB
- **Content**: Classroom video recordings with students visible

### Analysis Modes

#### Normal Mode
- Student detection and tracking
- Attention analysis (posture, eye tracking, head pose)
- Attendance tracking
- Distraction detection (phone usage, sleeping)

#### Exam Mode
- All Normal Mode features
- Enhanced cheating detection
- Strict attention monitoring
- Phone usage alerts
- Suspicious behavior tracking

### Legacy Command Line Mode
```bash
python run_monitoring.py
```

This launches an interactive menu with options:
1. **Start Regular Session** - Begin monitoring for regular class
2. **Start Exam Session** - Begin monitoring with cheating detection
3. **View Current Status** - Show system status and statistics
4. **View Student Analysis** - Show analysis for specific student
5. **View Attendance Report** - Show current session attendance
6. **Stop Monitoring** - End current session and stop monitoring
7. **Exit** - Exit the system

### Command Line Options
- `--auto-start`: Automatically start monitoring
- `--exam-mode`: Enable cheating detection
- `--session-type`: Session type (regular_class, exam, lecture, workshop)
- `--expected-students`: Expected number of students
- `--duration`: Monitoring duration in minutes (0 for infinite)

## 📊 System Analysis

### Attention Scoring
The system calculates attention scores based on multiple factors:
- **Eye Aspect Ratio (30%)**: Measures eye openness
- **Head Pose (25%)**: Tracks head orientation and movement
- **Sleeping Detection (20%)**: Identifies sleeping behavior
- **Phone Usage (15%)**: Detects phone usage patterns
- **Posture (10%)**: Analyzes body posture

### Distraction Detection
- **Phone Usage**: Detects hands near face area
- **Sleeping**: Combines eye closure and head position
- **Looking Away**: Excessive head movement detection
- **Poor Posture**: Uneven shoulders, head tilt

### Cheating Detection (Exam Mode)
- **Phone Usage**: High confidence cheating indicator
- **Excessive Head Movement**: Looking around suspiciously
- **Sleeping**: Inappropriate during exams
- **Low Attention**: Very low attention scores
- **Looking Down**: Frequent downward gaze

## 📈 Data Analysis & Reports

### Generated Reports
1. **Attendance Reports**: CSV files with student attendance data
2. **Attention Analysis**: Detailed attention scores and trends
3. **Session Summaries**: Overall session statistics
4. **Analysis Frames**: Annotated frames with detection results

### Data Aggregation
The system aggregates results over time windows:
- **Attention Windows**: 5-minute analysis periods
- **Attendance Windows**: 2-minute attendance validation
- **Frame Intervals**: 30-second processing intervals

### Example Output
```
Session Summary:
- Session ID: 12345-abcde
- Duration: 1:30:25
- Total Frames Analyzed: 180
- Students Detected: 25
- Average Attention Score: 0.78
- Average Cheating Probability: 0.12
```

## 🔧 Configuration Options

### Video Processing
```python
FRAME_INTERVAL = 30          # Process frame every 30 seconds
VIDEO_SOURCE = 0             # Camera index or video file path
FRAME_WIDTH = 1920           # Video width
FRAME_HEIGHT = 1080          # Video height
FPS = 30                     # Frames per second
```

### Face Recognition
```python
FACE_RECOGNITION_TOLERANCE = 0.6    # Recognition sensitivity
MIN_FACE_SIZE = 20                  # Minimum face size to process
FACE_DETECTION_CONFIDENCE = 0.5    # Detection confidence threshold
```

### Attention Analysis
```python
ATTENTION_THRESHOLD = 0.7           # High attention threshold
DISTRACTION_THRESHOLD = 0.3         # Distraction detection threshold
EYE_ASPECT_RATIO_THRESHOLD = 0.2    # Eye closure threshold
HEAD_POSE_THRESHOLD = 30            # Head movement threshold (degrees)
```

### Cheating Detection
```python
CHEATING_DETECTION_ENABLED = True   # Enable cheating detection
PHONE_DETECTION_CONFIDENCE = 0.7    # Phone detection confidence
SLEEPING_DETECTION_THRESHOLD = 0.8  # Sleeping detection threshold
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera Not Found**
   ```
   Error: Could not open video source
   ```
   **Solution**: Check camera connection and permissions. Try different camera indices (0, 1, 2).

2. **YOLO Model Not Found**
   ```
   Error loading YOLO model
   ```
   **Solution**: The system will automatically download YOLO models on first run.

3. **Face Recognition Training Issues**
   ```
   No images found for student
   ```
   **Solution**: Ensure dataset structure is correct and images contain clear faces.

4. **Performance Issues**
   ```
   Slow processing or high CPU usage
   ```
   **Solution**: 
   - Increase `FRAME_INTERVAL` to reduce processing frequency
   - Reduce video resolution in config
   - Use GPU acceleration if available

### Performance Optimization

1. **Reduce Frame Processing**
   ```python
   FRAME_INTERVAL = 60  # Process every 60 seconds instead of 30
   ```

2. **Lower Video Resolution**
   ```python
   FRAME_WIDTH = 1280
   FRAME_HEIGHT = 720
   ```

3. **Adjust Detection Thresholds**
   ```python
   PERSON_CONFIDENCE_THRESHOLD = 0.7  # Higher threshold for fewer detections
   ```

## 📝 Logging

The system generates detailed logs in the `logs/` directory:
- **monitoring.log**: Main system logs
- **CSV logs**: Detailed monitoring data in `data/monitoring_logs.csv`

### Log Levels
- **INFO**: General system information
- **WARNING**: Non-critical issues
- **ERROR**: System errors and failures
- **DEBUG**: Detailed debugging information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLO**: Real-time object detection
- **MediaPipe**: Face mesh and pose detection
- **OpenCV**: Computer vision processing
- **Face Recognition**: Face recognition library
- **Ultralytics**: YOLO implementation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/monitoring.log`
3. Create an issue with detailed error information

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with privacy laws and obtain necessary permissions before monitoring students.
