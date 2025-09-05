# 🚀 AI Student Monitoring System - Quick Start Guide

## ✅ System Status: WORKING!

Your AI Student Monitoring System is now fully functional! All AI models are working correctly:

- ✅ **MediaPipe**: Face detection and mesh analysis
- ✅ **YOLO**: Student/person detection
- ✅ **Face Recognition**: Student identification
- ✅ **Attention Analysis**: Attention and distraction detection
- ✅ **Web Interface**: Modern, responsive UI

## 🌐 How to Use the Web Interface

### 1. Start the Web Application
```bash
# Make sure you're in the project directory
cd newstudentmonitoringsystem

# Activate virtual environment
venv\Scripts\activate

# Start the web server
python app.py
```

### 2. Access the Web Interface
Open your browser and go to: **http://localhost:5000**

### 3. Upload and Analyze Videos

#### Step 1: Choose Analysis Mode
- **Normal Mode**: Regular classroom monitoring with attention tracking
- **Exam Mode**: Enhanced monitoring with cheating detection

#### Step 2: Upload Video File
- Supported formats: MP4, AVI, MOV, MKV, WMV
- Maximum size: 500MB
- Drag & drop or click to browse

#### Step 3: Monitor Progress
- Real-time progress bar
- Live status updates
- Processing information

#### Step 4: View Results
- Instant analysis report
- Student-by-student breakdown
- Download detailed reports
- View in report gallery

## 📊 What the System Analyzes

### Student Detection
- Detects multiple students (50+) using YOLO
- Tracks individual students across frames
- Estimates classroom density

### Face Recognition
- Identifies known students from your dataset
- Automatic attendance tracking
- Confidence scoring for recognition

### Attention Analysis
- **Eye tracking**: Monitors eye openness and movement
- **Head pose**: Tracks head orientation (yaw, pitch, roll)
- **Sleeping detection**: Identifies when students are sleeping
- **Phone usage**: Detects phone usage patterns
- **Posture analysis**: Monitors body posture

### Distraction Detection
- Phone usage alerts
- Sleeping behavior
- Looking away from screen/teacher
- Poor posture indicators

### Cheating Detection (Exam Mode)
- Enhanced phone usage monitoring
- Suspicious head movements
- Looking around behavior
- Very low attention scores

## 📁 Dataset Structure

Your dataset is already set up with 8 students:
```
dataset/
├── anush/          # 5 images
├── dhanush/        # 6 images
├── glen/           # 5 images
├── karthik/        # 4 images
├── leston/         # 11 images
└── ... (3 more students)
```

### Adding More Students
1. Create a folder with the student's name in `dataset/`
2. Add 5-10 clear photos of the student
3. Include different angles and expressions
4. Use JPG/PNG format

## 🎬 Testing the System

### Demo Video
A demo video (`demo_video.mp4`) has been created for testing. You can:
1. Upload it to the web interface
2. Choose any mode (Normal or Exam)
3. Watch the analysis progress
4. View the generated report

### Real Videos
For real classroom videos:
- Ensure good lighting
- Students should be clearly visible
- Camera should capture the entire classroom
- Avoid excessive movement or blur

## 📈 Understanding Reports

### Key Metrics
- **Students Detected**: Number of students identified
- **Avg Attention**: Overall class attention score
- **Frames Analyzed**: Number of frames processed
- **Cheating Risk**: Overall cheating probability

### Student Analysis
- **Attention Level**: Excellent/Good/Fair/Poor
- **Avg Attention**: Individual attention score
- **Cheating Risk**: Individual cheating probability
- **Detections**: Number of times detected

### Report Features
- **Download**: Get detailed JSON reports
- **Gallery**: Browse all previous reports
- **Timestamps**: Track when analysis was performed
- **Session Info**: Mode, duration, statistics

## 🔧 Troubleshooting

### Common Issues

1. **"No video file provided"**
   - Make sure you selected a video file
   - Check file format (MP4, AVI, MOV, MKV, WMV)

2. **"Could not open video file"**
   - Check if video file is corrupted
   - Try a different video format

3. **"Invalid video file"**
   - Ensure video has valid frames and FPS
   - Try a shorter video for testing

4. **Low detection rates**
   - Check video quality and lighting
   - Ensure students are clearly visible
   - Try adjusting camera angle

### Performance Tips

1. **For better accuracy**:
   - Use high-quality video recordings
   - Ensure good lighting conditions
   - Keep camera stable

2. **For faster processing**:
   - Use shorter video clips for testing
   - Reduce video resolution if needed
   - Close other applications

## 🎯 Next Steps

1. **Test with demo video**: Upload `demo_video.mp4` to see the system in action
2. **Add real student photos**: Replace sample dataset with actual student photos
3. **Record classroom videos**: Create test videos of your classroom
4. **Customize settings**: Adjust thresholds in `config.py` if needed

## 📞 Support

If you encounter any issues:
1. Check the console output for error messages
2. Ensure all dependencies are installed
3. Verify your dataset structure
4. Try the demo video first

---

**🎉 Congratulations! Your AI Student Monitoring System is ready to use!**

The system successfully combines:
- Advanced computer vision (YOLO + MediaPipe)
- Face recognition and attendance tracking
- Attention and distraction analysis
- Modern web interface
- Comprehensive reporting

You can now monitor student behavior, track attendance, and analyze classroom engagement with AI-powered insights!






