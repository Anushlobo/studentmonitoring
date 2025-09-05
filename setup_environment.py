#setup_environment.py
import subprocess
import sys
import os

def create_virtual_environment():
    """Create a virtual environment and install requirements"""
    try:
        # Create virtual environment
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
        # Determine the correct pip path
        if os.name == 'nt':  # Windows
            pip_path = "venv\\Scripts\\pip"
        else:  # Unix/Linux/Mac
            pip_path = "venv/bin/pip"
        
        # Upgrade pip
        print("Upgrading pip...")
        try:
            subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        except subprocess.CalledProcessError:
            # Try alternative method
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("Installing requirements...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Virtual environment created and dependencies installed successfully!")
        print("To activate the virtual environment:")
        if os.name == 'nt':
            print("venv\\Scripts\\activate")
        else:
            print("source venv/bin/activate")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_virtual_environment()
