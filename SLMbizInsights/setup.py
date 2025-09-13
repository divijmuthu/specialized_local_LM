#!/usr/bin/env python3
"""
SLM Business Insights System - Setup Script
Automates the initial setup process for first-time users
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🚀 SLM Business Insights System - Setup Script")
    print("=" * 60)
    print("This script will help you set up the system for the first time.")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("   Please install Python 3.8 or higher.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible.")
    return True

def check_pip():
    """Check if pip is available"""
    print("🔍 Checking pip...")
    
    try:
        import pip
        print("✅ pip is available.")
        return True
    except ImportError:
        print("❌ pip is not available.")
        print("   Please install pip.")
        return False

def create_directories():
    """Create required directories"""
    print("📁 Creating required directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models/saved",
        "models/trained",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    print("✅ All directories created successfully.")

def create_virtual_environment():
    """Create virtual environment"""
    print("🐍 Creating virtual environment...")
    
    try:
        # Check if virtual environment already exists
        if Path("slm_env").exists():
            print("   ⚠️  Virtual environment already exists.")
            response = input("   Do you want to recreate it? (y/N): ").lower()
            if response == 'y':
                import shutil
                shutil.rmtree("slm_env")
            else:
                print("   ✅ Using existing virtual environment.")
                return True
        
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "slm_env"], check=True)
        print("   ✅ Virtual environment created successfully.")
        return True
        
    except subprocess.CalledProcessError:
        print("   ❌ Failed to create virtual environment.")
        return False

def get_activation_command():
    """Get the correct activation command for the platform"""
    if platform.system() == "Windows":
        return "slm_env\\Scripts\\activate"
    else:
        return "source slm_env/bin/activate"

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Get the correct pip path
    if platform.system() == "Windows":
        pip_path = "slm_env\\Scripts\\pip"
    else:
        pip_path = "slm_env/bin/pip"
    
    try:
        # Upgrade pip first
        print("   🔄 Upgrading pip...")
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("   📦 Installing requirements...")
        try:
            subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        except subprocess.CalledProcessError:
            print("   ⚠️  Full requirements failed, trying minimal requirements...")
            subprocess.run([pip_path, "install", "-r", "requirements-minimal.txt"], check=True)
        
        print("   ✅ Dependencies installed successfully.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        return False

def test_installation():
    """Test if the installation is working"""
    print("🧪 Testing installation...")
    
    # Get the correct python path
    if platform.system() == "Windows":
        python_path = "slm_env\\Scripts\\python"
    else:
        python_path = "slm_env/bin/python"
    
    try:
        # Test basic imports
        test_script = """
import sys
print('Python version:', sys.version)

try:
    import pandas, numpy, sklearn, flask, streamlit
    print('✅ All core dependencies imported successfully!')
    print('✅ Installation test passed!')
except ImportError as e:
    print('❌ Import error:', e)
    sys.exit(1)
"""
        
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Installation test passed!")
            print("   📊 Test output:")
            for line in result.stdout.strip().split('\n'):
                print(f"      {line}")
            return True
        else:
            print("   ❌ Installation test failed!")
            print("   📊 Error output:")
            for line in result.stderr.strip().split('\n'):
                print(f"      {line}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to run installation test: {e}")
        return False

def run_simple_test():
    """Run the simple test to verify everything works"""
    print("🧪 Running system test...")
    
    # Get the correct python path
    if platform.system() == "Windows":
        python_path = "slm_env\\Scripts\\python"
    else:
        python_path = "slm_env/bin/python"
    
    try:
        result = subprocess.run([python_path, "simple_test.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ System test passed!")
            print("   📊 Test summary:")
            # Extract test results from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Test Results:' in line or 'ALL TESTS PASSED' in line:
                    print(f"      {line}")
            return True
        else:
            print("   ❌ System test failed!")
            print("   📊 Error output:")
            for line in result.stderr.strip().split('\n')[:10]:  # Show first 10 lines
                print(f"      {line}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to run system test: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    
    activation_cmd = get_activation_command()
    
    print("\n📋 Next Steps:")
    print("1. Activate the virtual environment:")
    print(f"   {activation_cmd}")
    print()
    print("2. Run the complete system:")
    print("   python3 app.py --mode all")
    print()
    print("3. Access the dashboard:")
    print("   http://localhost:8501")
    print()
    print("4. Access the API:")
    print("   http://localhost:5000")
    print()
    print("5. Run tests:")
    print("   python3 simple_test.py")
    print()
    print("📚 For detailed instructions, see GETTING_STARTED.md")
    print()
    print("🚀 Happy analyzing!")

def main():
    """Main setup function"""
    print_header()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n⚠️  Installation test failed, but you can still try running the system.")
        print("   Some optional dependencies might be missing.")
    
    # Run simple test
    print("\n🧪 Running system test...")
    if not run_simple_test():
        print("\n⚠️  System test failed, but the basic installation is complete.")
        print("   You can still try running the system manually.")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
