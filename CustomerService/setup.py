#!/usr/bin/env python3
"""
Customer Service Feedback Analysis Tool - Setup Script
Automated setup for first-time users
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🎧 Customer Service Feedback Analysis Tool - Setup")
    print("=" * 60)
    print("This script will set up your environment for the Customer Service")
    print("Feedback Analysis Tool with AI-powered insights.")
    print()

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print("\n🔧 Creating virtual environment...")
    
    venv_name = "customer_service_env"
    
    if os.path.exists(venv_name):
        print(f"✅ Virtual environment '{venv_name}' already exists")
        return venv_name
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print(f"✅ Virtual environment '{venv_name}' created successfully")
        return venv_name
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return None

def get_pip_path(venv_name):
    """Get pip path for virtual environment"""
    if platform.system() == "Windows":
        return os.path.join(venv_name, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_name, "bin", "pip")

def install_dependencies(venv_name):
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    pip_path = get_pip_path(venv_name)
    
    if not os.path.exists(pip_path):
        print(f"❌ Pip not found at {pip_path}")
        return False
    
    try:
        # Upgrade pip first
        print("   🔄 Upgrading pip...")
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("   📦 Installing requirements...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        print("   ✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "uploads",
        "saved_model",
        "logs",
        "results",
        "static",
        "templates"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✅ Created directory: {directory}")
        else:
            print(f"   ✅ Directory already exists: {directory}")

def download_nltk_data():
    """Download NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("   ✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"   ⚠️  NLTK data download failed: {e}")
        print("   ℹ️  This will be handled automatically when the app runs")
        return True

def test_installation(venv_name):
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    python_path = os.path.join(venv_name, "bin", "python") if platform.system() != "Windows" else os.path.join(venv_name, "Scripts", "python.exe")
    
    if not os.path.exists(python_path):
        print(f"❌ Python executable not found at {python_path}")
        return False
    
    try:
        # Test imports
        test_script = """
import sys
try:
    import flask
    import pandas
    import numpy
    import sklearn
    import nltk
    import textblob
    print("✅ All core dependencies imported successfully")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
        
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Installation test passed")
            return True
        else:
            print(f"   ❌ Installation test failed: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"   ❌ Installation test error: {e}")
        return False

def create_sample_data():
    """Create sample data file"""
    print("\n📄 Creating sample data...")
    
    sample_data = """text,date,source
"I love your product! It's amazing and works perfectly.",2023-01-01,email
"Customer service was terrible. They didn't respond to my issue.",2023-01-02,review
"The product is good, but the pricing is too high.",2023-01-03,survey
"Excellent support team! They resolved my issue quickly.",2023-01-04,email
"The new feature is great! It solves a major problem for us.",2023-01-05,survey
"I've been a customer for years, but recent changes are disappointing.",2023-01-06,social_media
"Fast delivery and good quality product. Highly recommend!",2023-01-07,review
"Poor quality product. Not worth the money.",2023-01-08,review
"The support team was very helpful in resolving my issue.",2023-01-09,email
"The product works well, but the documentation is lacking.",2023-01-10,survey
"""
    
    try:
        with open("sample_feedback.csv", "w") as f:
            f.write(sample_data)
        print("   ✅ Sample data file created: sample_feedback.csv")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create sample data: {e}")
        return False

def print_completion_message(venv_name):
    """Print completion message"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Your Customer Service Feedback Analysis Tool is ready to use!")
    print()
    print("📋 Next Steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print(f"   {venv_name}\\Scripts\\activate")
    else:
        print(f"   source {venv_name}/bin/activate")
    print()
    print("2. Run the application:")
    print("   python app.py")
    print()
    print("3. Open your browser and go to:")
    print("   http://localhost:5000")
    print()
    print("4. Upload the sample data file (sample_feedback.csv) to test the system")
    print()
    print("📚 Features Available:")
    print("   ✅ Sentiment Analysis (Positive/Negative/Neutral)")
    print("   ✅ Topic Modeling (LDA)")
    print("   ✅ Key Phrase Extraction")
    print("   ✅ Actionable Insights Generation")
    print("   ✅ Trend Analysis")
    print("   ✅ Interactive Dashboard")
    print("   ✅ File Upload (CSV/Excel)")
    print("   ✅ Model Training")
    print()
    print("🔧 Troubleshooting:")
    print("   - If you encounter issues, check the logs in the 'logs' directory")
    print("   - Make sure all dependencies are installed correctly")
    print("   - For support, check the documentation or contact the development team")
    print()
    print("🚀 Ready to analyze customer feedback with AI-powered insights!")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    venv_name = create_virtual_environment()
    if not venv_name:
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(venv_name):
        print("\n⚠️  Dependency installation failed. You may need to install them manually.")
        print("   Try running: pip install -r requirements.txt")
    
    # Create directories
    create_directories()
    
    # Download NLTK data
    download_nltk_data()
    
    # Test installation
    if not test_installation(venv_name):
        print("\n⚠️  Installation test failed. The application may not work correctly.")
    
    # Create sample data
    create_sample_data()
    
    # Print completion message
    print_completion_message(venv_name)

if __name__ == "__main__":
    main()
