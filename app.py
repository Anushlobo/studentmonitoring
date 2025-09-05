from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import cv2
import numpy as np
import json
import pandas as pd
from datetime import datetime
import uuid
import threading
import time
from werkzeug.utils import secure_filename
import logging

from config import *
from database import MonitoringDatabase
from face_recognition_module import FaceRecognitionModule
from attention_analysis import AttentionAnalysisModule
from student_detection import StudentDetectionModule
from main_monitoring_system import StudentMonitoringSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = 'reports'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
os.makedirs('static/reports', exist_ok=True)

# Global variables
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
current_session = None
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path, mode, session_id):
    """Process video file and generate reports"""
    global processing_status
    
    try:
        processing_status[session_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing video processing...'
        }
        
        # Initialize monitoring system
        monitoring_system = StudentMonitoringSystem()
        
        # Start session
        session_type = 'exam' if mode == 'exam' else 'regular_class'
        monitoring_system.start_session(session_type, mode == 'exam', 0)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            processing_status[session_id]['status'] = 'error'
            processing_status[session_id]['message'] = 'Could not open video file'
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Ensure we have valid values
        if total_frames <= 0 or fps <= 0:
            processing_status[session_id]['status'] = 'error'
            processing_status[session_id]['message'] = 'Invalid video file (no frames or fps)'
            cap.release()
            return
        
        # Calculate video duration
        video_duration = total_frames / fps
        
        # Calculate frame interval for 30-second processing
        frame_interval = max(1, int(fps * FRAME_INTERVAL))  # Process every 30 seconds
        
        # For videos shorter than 30 seconds, ensure we process at least 1 frame
        if video_duration < FRAME_INTERVAL:
            # For short videos, process the middle frame
            middle_frame = total_frames // 2
            frame_interval = middle_frame if middle_frame > 0 else 1
            processing_status[session_id]['message'] = f'Short video detected ({video_duration:.1f}s). Processing middle frame.'
        else:
            processing_status[session_id]['message'] = f'Processing video: {total_frames} frames, {fps:.1f} fps (every {FRAME_INTERVAL}s)'
        
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frames based on interval
            if frame_count % frame_interval == 0:
                try:
                    # Process frame (bypass interval check since we're already filtering frames)
                    results = monitoring_system.process_frame(frame, bypass_interval=True)
                    processed_frames += 1
                    
                    # Update progress
                    progress = (frame_count / total_frames) * 100
                    processing_status[session_id]['progress'] = progress
                    
                    if video_duration < FRAME_INTERVAL:
                        processing_status[session_id]['message'] = f'Short video: Processed {processed_frames} frame ({progress:.1f}%)'
                    else:
                        processing_status[session_id]['message'] = f'Processed {processed_frames} analysis frames ({progress:.1f}%)'
                    
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
        
        cap.release()
        
        # Generate reports
        processing_status[session_id]['message'] = 'Generating reports...'
        monitoring_system.end_session()
        
        # Generate comprehensive report
        report_data = generate_comprehensive_report(monitoring_system, session_id)
        
        # Save report
        report_filename = f"report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        processing_status[session_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Analysis completed successfully!',
            'report_file': report_filename,
            'report_data': report_data
        }
        
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        processing_status[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

def generate_comprehensive_report(monitoring_system, session_id):
    """Generate comprehensive analysis report"""
    
    # Get session data
    session_duration = str(datetime.now() - monitoring_system.session_start_time) if monitoring_system.session_start_time else "N/A"
    total_frames = len(monitoring_system.frame_analysis_results)
    total_students = len(monitoring_system.attendance_tracking)
    
    # Calculate statistics
    all_attention_scores = []
    all_cheating_probs = []
    student_analyses = {}
    
    for frame_result in monitoring_system.frame_analysis_results:
        for result in frame_result['results']:
            all_attention_scores.append(result['attention_score'])
            all_cheating_probs.append(result['cheating_probability'])
            
            # Track individual student data
            student_id = result['student_id']
            if student_id not in student_analyses:
                student_analyses[student_id] = {
                    'attention_scores': [],
                    'cheating_probs': [],
                    'detection_count': 0,
                    'first_seen': result['timestamp'],
                    'last_seen': result['timestamp']
                }
            
            student_analyses[student_id]['attention_scores'].append(result['attention_score'])
            student_analyses[student_id]['cheating_probs'].append(result['cheating_probability'])
            student_analyses[student_id]['detection_count'] += 1
            student_analyses[student_id]['last_seen'] = result['timestamp']
    
    # Calculate averages
    avg_attention = sum(all_attention_scores) / len(all_attention_scores) if all_attention_scores else 0
    avg_cheating_prob = sum(all_cheating_probs) / len(all_cheating_probs) if all_cheating_probs else 0
    
    # Generate student summaries
    student_summaries = {}
    for student_id, data in student_analyses.items():
        if data['attention_scores']:
            avg_student_attention = sum(data['attention_scores']) / len(data['attention_scores'])
            avg_student_cheating = sum(data['cheating_probs']) / len(data['cheating_probs'])
            
            # Determine attention level
            if avg_student_attention > 0.8:
                attention_level = "Excellent"
            elif avg_student_attention > 0.6:
                attention_level = "Good"
            elif avg_student_attention > 0.4:
                attention_level = "Fair"
            else:
                attention_level = "Poor"
            
            student_summaries[student_id] = {
                'average_attention': avg_student_attention,
                'average_cheating_probability': avg_student_cheating,
                'attention_level': attention_level,
                'detection_count': data['detection_count'],
                'first_seen': data['first_seen'],
                'last_seen': data['last_seen']
            }
    
    # Create comprehensive report
    report = {
        'session_id': session_id,
        'session_type': 'exam' if monitoring_system.exam_mode else 'regular',
        'session_duration': session_duration,
        'total_frames_analyzed': total_frames,
        'total_students_detected': total_students,
        'overall_statistics': {
            'average_attention_score': avg_attention,
            'average_cheating_probability': avg_cheating_prob,
            'total_analysis_frames': total_frames
        },
        'student_analyses': student_summaries,
        'attendance_data': monitoring_system.attendance_tracking,
        'frame_analysis_results': monitoring_system.frame_analysis_results,
        'generated_at': datetime.now().isoformat()
    }
    
    return report

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    mode = request.form.get('mode', 'normal')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload MP4, AVI, MOV, MKV, or WMV files.'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    session_id = str(uuid.uuid4())
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
    file.save(video_path)
    
    # Start processing in background thread
    thread = threading.Thread(target=process_video, args=(video_path, mode, session_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'session_id': session_id,
        'message': 'Video uploaded successfully. Processing started.',
        'status': 'processing'
    })

@app.route('/status/<session_id>')
def get_status(session_id):
    if session_id in processing_status:
        return jsonify(processing_status[session_id])
    else:
        return jsonify({'status': 'not_found', 'message': 'Session not found'})

@app.route('/report/<session_id>')
def get_report(session_id):
    if session_id in processing_status and processing_status[session_id]['status'] == 'completed':
        return jsonify(processing_status[session_id]['report_data'])
    else:
        return jsonify({'error': 'Report not available'}), 404

@app.route('/download/<filename>')
def download_report(filename):
    try:
        return send_file(
            os.path.join(app.config['REPORTS_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        return jsonify({'error': 'Report file not found'}), 404

@app.route('/gallery')
def gallery():
    """Show all generated reports"""
    reports = []
    for filename in os.listdir(app.config['REPORTS_FOLDER']):
        if filename.endswith('.json'):
            filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
            try:
                with open(filepath, 'r') as f:
                    report_data = json.load(f)
                
                reports.append({
                    'filename': filename,
                    'session_id': report_data.get('session_id', 'Unknown'),
                    'session_type': report_data.get('session_type', 'Unknown'),
                    'session_duration': report_data.get('session_duration', 'Unknown'),
                    'total_students': report_data.get('total_students_detected', 0),
                    'average_attention': report_data.get('overall_statistics', {}).get('average_attention_score', 0),
                    'generated_at': report_data.get('generated_at', 'Unknown')
                })
            except Exception as e:
                logging.error(f"Error reading report {filename}: {e}")
    
    # Sort by generation date (newest first)
    reports.sort(key=lambda x: x['generated_at'], reverse=True)
    
    return render_template('gallery.html', reports=reports)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
