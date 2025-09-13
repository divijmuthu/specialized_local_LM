"""
Run the Streamlit dashboard
"""
import subprocess
import sys
import os

def run_dashboard():
    """Run the Streamlit dashboard"""
    dashboard_path = os.path.join(os.path.dirname(__file__), 'ui', 'dashboard.py')
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', dashboard_path,
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    run_dashboard()
