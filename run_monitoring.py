#!/usr/bin/env python3
#run_monitoring.py
"""
AI-Based Student Monitoring System
Main execution script
"""

import sys
import time
import signal
import argparse
import logging
from datetime import datetime

from main_monitoring_system import StudentMonitoringSystem
from database import MonitoringDatabase

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\n🛑 Received interrupt signal. Shutting down gracefully...")
    if hasattr(signal_handler, 'monitoring_system'):
        signal_handler.monitoring_system.stop_monitoring()
    sys.exit(0)

def print_banner():
    """Print system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           AI-Based Student Monitoring System                 ║
    ║                                                              ║
    ║  Features:                                                   ║
    ║  • Multi-student detection (50+ students)                   ║
    ║  • Face recognition & attendance tracking                   ║
    ║  • Attention analysis (posture, eye tracking, head pose)    ║
    ║  • Distraction detection (phone usage, sleeping)           ║
    ║  • Cheating detection (exam mode)                          ║
    ║  • Frame analysis every 30 seconds                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print main menu options"""
    menu = """
    📋 Available Commands:
    
    1. Start Regular Session    - Begin monitoring for regular class
    2. Start Exam Session       - Begin monitoring with cheating detection
    3. View Current Status      - Show system status and statistics
    4. View Student Analysis    - Show analysis for specific student
    5. View Attendance Report   - Show current session attendance
    6. Stop Monitoring          - End current session and stop monitoring
    7. Exit                     - Exit the system
    
    Enter your choice (1-7): """
    return input(menu)

def start_session_interactive(monitoring_system):
    """Interactive session start"""
    print("\n🎯 Starting New Session")
    print("=" * 40)
    
    # Session type
    print("Session Types:")
    print("1. Regular Class")
    print("2. Exam/Test")
    print("3. Lecture")
    print("4. Workshop")
    
    session_choice = input("Select session type (1-4): ").strip()
    session_types = {
        '1': 'regular_class',
        '2': 'exam',
        '3': 'lecture',
        '4': 'workshop'
    }
    session_type = session_types.get(session_choice, 'regular_class')
    
    # Exam mode
    exam_mode = session_type == 'exam'
    
    # Expected students
    try:
        expected_students = int(input("Expected number of students (0 for unknown): ").strip() or "0")
    except ValueError:
        expected_students = 0
    
    # Start session
    session_id = monitoring_system.start_session(session_type, exam_mode, expected_students)
    
    if session_id:
        print(f"✅ Session started successfully!")
        print(f"Session ID: {session_id}")
        print(f"Session Type: {session_type}")
        print(f"Exam Mode: {exam_mode}")
        print(f"Expected Students: {expected_students}")
        
        # Start monitoring
        if monitoring_system.start_monitoring():
            print("🎥 Monitoring started successfully!")
            print("📊 System is now analyzing frames every 30 seconds...")
            return True
        else:
            print("❌ Failed to start monitoring")
            return False
    else:
        print("❌ Failed to start session")
        return False

def view_status(monitoring_system):
    """View current system status"""
    print("\n📊 System Status")
    print("=" * 40)
    
    status = monitoring_system.get_current_status()
    
    if not status:
        print("❌ Unable to retrieve system status")
        return
    
    print(f"🔄 Running: {'Yes' if status['is_running'] else 'No'}")
    print(f"🆔 Session ID: {status['session_id'] or 'None'}")
    print(f"📹 Frames Processed: {status['frame_count']}")
    print(f"👥 Students Tracked: {status['total_students_tracked']}")
    print(f"⏱️  Session Duration: {status['session_duration']}")
    print(f"📝 Exam Mode: {'Yes' if status['exam_mode'] else 'No'}")
    
    # Show recent activity
    if monitoring_system.frame_analysis_results:
        recent_results = monitoring_system.frame_analysis_results[-5:]  # Last 5 frames
        print(f"\n📈 Recent Activity (Last {len(recent_results)} frames):")
        for result in recent_results:
            timestamp = result['timestamp'].strftime("%H:%M:%S")
            students = result['total_students']
            print(f"  {timestamp}: {students} students detected")

def view_student_analysis(monitoring_system):
    """View analysis for a specific student"""
    print("\n👤 Student Analysis")
    print("=" * 40)
    
    # Get list of tracked students
    tracked_students = list(monitoring_system.attendance_tracking.keys())
    
    if not tracked_students:
        print("❌ No students have been tracked yet")
        return
    
    print("Tracked Students:")
    for i, student_id in enumerate(tracked_students, 1):
        print(f"{i}. {student_id}")
    
    try:
        choice = int(input(f"\nSelect student (1-{len(tracked_students)}): ").strip())
        if 1 <= choice <= len(tracked_students):
            student_id = tracked_students[choice - 1]
        else:
            print("❌ Invalid choice")
            return
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Get analysis data
    analysis_data = monitoring_system.get_student_analysis(student_id)
    
    if not analysis_data:
        print(f"❌ No analysis data available for {student_id}")
        return
    
    print(f"\n📊 Analysis for {student_id}:")
    print(f"Total observations: {len(analysis_data)}")
    
    # Calculate averages
    attention_scores = [data['attention_score'] for data in analysis_data]
    distraction_scores = [data['distraction_score'] for data in analysis_data]
    cheating_probs = [data['cheating_probability'] for data in analysis_data]
    
    if attention_scores:
        avg_attention = sum(attention_scores) / len(attention_scores)
        avg_distraction = sum(distraction_scores) / len(distraction_scores)
        avg_cheating = sum(cheating_probs) / len(cheating_probs)
        
        print(f"Average Attention Score: {avg_attention:.3f}")
        print(f"Average Distraction Score: {avg_distraction:.3f}")
        print(f"Average Cheating Probability: {avg_cheating:.3f}")
        
        # Attention level
        if avg_attention > 0.8:
            attention_level = "Excellent"
        elif avg_attention > 0.6:
            attention_level = "Good"
        elif avg_attention > 0.4:
            attention_level = "Fair"
        else:
            attention_level = "Poor"
        
        print(f"Overall Attention Level: {attention_level}")
    
    # Show recent observations
    print(f"\n📝 Recent Observations:")
    for data in analysis_data[-3:]:  # Last 3 observations
        timestamp = data['timestamp'].strftime("%H:%M:%S")
        attention = data['attention_score']
        distraction = data['distraction_score']
        cheating = data['cheating_probability']
        
        print(f"  {timestamp}: Attention={attention:.2f}, Distraction={distraction:.2f}, Cheating={cheating:.2f}")

def view_attendance_report(monitoring_system):
    """View attendance report"""
    print("\n📋 Attendance Report")
    print("=" * 40)
    
    if not monitoring_system.session_id:
        print("❌ No active session")
        return
    
    attendance_data = monitoring_system.attendance_tracking
    
    if not attendance_data:
        print("❌ No attendance data available")
        return
    
    print(f"Session: {monitoring_system.session_id}")
    print(f"Total Students Present: {len(attendance_data)}")
    print(f"Session Duration: {monitoring_system.session_start_time and str(datetime.now() - monitoring_system.session_start_time)}")
    
    print(f"\n📊 Student Attendance:")
    print(f"{'Student ID':<15} {'First Seen':<12} {'Last Seen':<12} {'Confidence':<10}")
    print("-" * 55)
    
    for student_id, data in attendance_data.items():
        first_seen = data['first_seen'].strftime("%H:%M:%S")
        last_seen = data['last_seen'].strftime("%H:%M:%S")
        confidence = data['confidence']
        
        print(f"{student_id:<15} {first_seen:<12} {last_seen:<12} {confidence:<10.3f}")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI-Based Student Monitoring System')
    parser.add_argument('--auto-start', action='store_true', help='Automatically start monitoring')
    parser.add_argument('--exam-mode', action='store_true', help='Start in exam mode')
    parser.add_argument('--session-type', default='regular_class', help='Session type')
    parser.add_argument('--expected-students', type=int, default=0, help='Expected number of students')
    parser.add_argument('--duration', type=int, default=0, help='Monitoring duration in minutes (0 for infinite)')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize monitoring system
    print("🔧 Initializing monitoring system...")
    monitoring_system = StudentMonitoringSystem()
    signal_handler.monitoring_system = monitoring_system
    
    try:
        if args.auto_start:
            # Auto-start mode
            print("🚀 Auto-start mode enabled")
            session_id = monitoring_system.start_session(
                args.session_type, 
                args.exam_mode, 
                args.expected_students
            )
            
            if session_id and monitoring_system.start_monitoring():
                print("✅ Monitoring started automatically")
                print("Press Ctrl+C to stop")
                
                # Run for specified duration
                if args.duration > 0:
                    print(f"⏰ Running for {args.duration} minutes...")
                    time.sleep(args.duration * 60)
                    monitoring_system.stop_monitoring()
                else:
                    # Run indefinitely
                    while monitoring_system.is_running:
                        time.sleep(1)
            else:
                print("❌ Failed to start monitoring automatically")
                return 1
        else:
            # Interactive mode
            print("🎮 Interactive mode")
            print("Type 'help' for available commands")
            
            while True:
                try:
                    choice = print_menu()
                    
                    if choice == '1':
                        # Start regular session
                        if not monitoring_system.is_running:
                            start_session_interactive(monitoring_system)
                        else:
                            print("❌ Monitoring is already running")
                    
                    elif choice == '2':
                        # Start exam session
                        if not monitoring_system.is_running:
                            session_id = monitoring_system.start_session('exam', True, 0)
                            if session_id and monitoring_system.start_monitoring():
                                print("✅ Exam session started with cheating detection!")
                            else:
                                print("❌ Failed to start exam session")
                        else:
                            print("❌ Monitoring is already running")
                    
                    elif choice == '3':
                        # View status
                        view_status(monitoring_system)
                    
                    elif choice == '4':
                        # View student analysis
                        view_student_analysis(monitoring_system)
                    
                    elif choice == '5':
                        # View attendance report
                        view_attendance_report(monitoring_system)
                    
                    elif choice == '6':
                        # Stop monitoring
                        if monitoring_system.is_running:
                            monitoring_system.stop_monitoring()
                            print("✅ Monitoring stopped")
                        else:
                            print("❌ No monitoring session is running")
                    
                    elif choice == '7':
                        # Exit
                        if monitoring_system.is_running:
                            monitoring_system.stop_monitoring()
                        print("👋 Goodbye!")
                        break
                    
                    else:
                        print("❌ Invalid choice. Please select 1-7.")
                    
                    print("\n" + "="*50 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n🛑 Interrupted by user")
                    if monitoring_system.is_running:
                        monitoring_system.stop_monitoring()
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    logging.error(f"Error in main loop: {e}")
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}")
        return 1
    
    finally:
        # Cleanup
        if monitoring_system.is_running:
            monitoring_system.stop_monitoring()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

