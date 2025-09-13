# 🚀 SLM Business Insights System - Getting Started Guide

Welcome to the **Small Language Model for Business Insights (SLM-BI)** system! This guide will walk you through setting up and running the system for the first time.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Testing the System](#testing-the-system)
6. [Using the Dashboard](#using-the-dashboard)
7. [API Usage](#api-usage)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

## 🔧 Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required Software
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **pip** (Python package installer)
- **Git** (for version control)

### System Requirements
- **RAM**: Minimum 8GB (recommended: 16GB+)
- **Storage**: At least 5GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Check Your Installation
```bash
# Check Python version
python3 --version

# Check pip version
pip3 --version

# Check Git version
git --version
```

## 📦 Installation

### Step 1: Clone or Navigate to the Project
```bash
# If you have the project in a Git repository
git clone <repository-url>
cd SLMbizInsights

# Or if you already have the project locally
cd /path/to/SLMbizInsights
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv slm_env

# Activate virtual environment
# On macOS/Linux:
source slm_env/bin/activate

# On Windows:
# slm_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Option 1: Install all required packages (recommended)
pip install -r requirements.txt

# Option 2: Install minimal requirements (if full installation fails)
pip install -r requirements-minimal.txt

# This will install:
# - Core ML libraries (scikit-learn, pandas, numpy)
# - Deep learning frameworks (transformers, torch) - full version only
# - API framework (Flask)
# - Dashboard framework (Streamlit)
# - Data processing tools
# - Security and monitoring libraries
```

### Step 4: Verify Installation
```bash
# Run a quick test to verify everything is installed correctly
python3 -c "
import sys
print('Python version:', sys.version)
try:
    import pandas, numpy, sklearn, flask, streamlit
    print('✅ All core dependencies installed successfully!')
except ImportError as e:
    print('❌ Missing dependency:', e)
"
```

## ⚙️ Configuration

### Step 1: Review Configuration Settings
The system uses `config.py` for configuration. Key settings include:

```python
# Database settings
DATABASE_URL = "sqlite:///data/business_insights.db"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000

# Dashboard settings
DASHBOARD_PORT = 8501

# Model settings
MODEL_CACHE_DIR = "models/saved"
```

### Step 2: Create Required Directories
```bash
# Create data directories
mkdir -p data/raw data/processed data/external
mkdir -p models/saved models/trained
mkdir -p logs
```

### Step 3: Set Environment Variables (Optional)
```bash
# Set API key for external data sources (if needed)
export OPENAI_API_KEY="your-api-key-here"
export GOOGLE_API_KEY="your-google-api-key-here"

# Set database URL (if using external database)
export DATABASE_URL="postgresql://user:password@localhost/dbname"
```

## 🚀 Running the System

### Option 1: Run Everything (Recommended for First Time)
```bash
# Run the complete system (API + Dashboard + Pipeline)
python3 app.py --mode all

# This will start:
# - Flask API server on http://localhost:5000
# - Streamlit dashboard on http://localhost:8501
# - Data processing pipeline
# - Model training pipeline
```

### Option 2: Run Individual Components

#### Run API Only
```bash
python3 app.py --mode api
# Access API at: http://localhost:5000
# Interactive docs at: http://localhost:5000
```

#### Run Dashboard Only
```bash
python3 app.py --mode dashboard
# Access dashboard at: http://localhost:8501
```

#### Run Pipeline Only
```bash
python3 app.py --mode pipeline
# Processes data and generates insights
```

#### Run Training Only
```bash
python3 app.py --mode training
# Trains all ML models
```

### Option 3: Run with Custom Configuration
```bash
# Run with custom settings
python3 app.py --mode all --host 0.0.0.0 --port 5000 --dashboard-port 8501
```

## 🧪 Testing the System

### Step 1: Run the Simple Test
```bash
# Run the comprehensive test with sample data
python3 simple_test.py

# This will test:
# - Data generation
# - Customer segmentation
# - Sentiment analysis
# - Sales forecasting
# - Price optimization
# - Churn prediction
# - Recommendation engine
# - Data anonymization
```

### Step 2: Run the Full Test Suite
```bash
# Run all tests
python3 run_tests.py

# This will run:
# - Unit tests for all components
# - Integration tests
# - API endpoint tests
# - Security tests
# - Performance tests
```

### Step 3: Test API Endpoints
```bash
# Test API health
curl http://localhost:5000/health

# Test sample data generation
curl http://localhost:5000/api/data/sample

# Test customer segmentation
curl -X POST http://localhost:5000/api/insights/customer-segmentation \
  -H "Content-Type: application/json" \
  -d '{"customers": [{"customer_id": "CUST_001", "age": 25, "total_spend": 1000}], "n_clusters": 3}'
```

## 📊 Using the Dashboard

### Step 1: Access the Dashboard
1. Open your web browser
2. Navigate to `http://localhost:8501`
3. You should see the SLM Business Insights dashboard

### Step 2: Navigate the Dashboard
The dashboard includes several sections:

#### 📈 Overview Tab
- System health status
- Key performance indicators
- Recent insights summary

#### 👥 Customer Insights Tab
- Customer segmentation analysis
- Churn prediction results
- Customer lifetime value

#### 💰 Sales & Revenue Tab
- Sales forecasting
- Price optimization recommendations
- Revenue analysis

#### 😊 Sentiment Analysis Tab
- Customer feedback analysis
- Sentiment trends
- Text insights

#### 💡 Recommendations Tab
- Marketing recommendations
- Pricing strategies
- Supply chain optimization

#### ⚙️ Settings Tab
- Model configuration
- Data source settings
- System preferences

### Step 3: Generate Insights
1. Click on any tab to explore insights
2. Use the "Generate New Insights" button to run analysis
3. View results in interactive charts and tables
4. Export insights as CSV or PDF

## 🌐 API Usage

### Step 1: Access API Documentation
1. Open your web browser
2. Navigate to `http://localhost:5000`
3. You'll see the interactive API documentation

### Step 2: Test API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Generate Sample Data
```bash
curl http://localhost:5000/api/data/sample
```

#### Customer Segmentation
```bash
curl -X POST http://localhost:5000/api/insights/customer-segmentation \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"customer_id": "CUST_001", "age": 25, "total_spend": 1000, "purchase_count": 5},
      {"customer_id": "CUST_002", "age": 35, "total_spend": 2000, "purchase_count": 10}
    ],
    "n_clusters": 3
  }'
```

#### Sentiment Analysis
```bash
curl -X POST http://localhost:5000/api/insights/sentiment-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great product, very satisfied!",
      "Poor quality, disappointed"
    ]
  }'
```

#### Sales Forecasting
```bash
curl -X POST http://localhost:5000/api/insights/sales-forecasting \
  -H "Content-Type: application/json" \
  -d '{
    "sales_data": [
      {"date": "2023-01-01", "sales": 1000},
      {"date": "2023-02-01", "sales": 1100}
    ],
    "periods": 7
  }'
```

### Step 3: Use API in Your Applications
```python
import requests

# Example: Get customer segmentation
response = requests.post('http://localhost:5000/api/insights/customer-segmentation', 
                        json={
                            'customers': customer_data,
                            'n_clusters': 5
                        })
segments = response.json()
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "Module not found" errors
```bash
# Solution: Ensure virtual environment is activated and dependencies are installed
source slm_env/bin/activate  # or slm_env\Scripts\activate on Windows
pip install -r requirements.txt
```

#### Issue: Port already in use
```bash
# Solution: Use different ports
python3 app.py --mode all --port 5001 --dashboard-port 8502
```

#### Issue: Memory errors during model training
```bash
# Solution: Reduce batch size or use smaller models
# Edit config.py to set smaller batch sizes
BATCH_SIZE = 32  # Reduce from default 64
```

#### Issue: API not responding
```bash
# Solution: Check if the service is running
curl http://localhost:5000/health

# If not responding, restart the service
python3 app.py --mode api
```

#### Issue: Dashboard not loading
```bash
# Solution: Check if Streamlit is running
# Restart the dashboard
python3 app.py --mode dashboard
```

### Getting Help

#### Check Logs
```bash
# View system logs
tail -f logs/system.log

# View API logs
tail -f logs/api.log

# View dashboard logs
tail -f logs/dashboard.log
```

#### Debug Mode
```bash
# Run in debug mode for detailed error messages
python3 app.py --mode all --debug
```

#### System Status
```bash
# Check system status
python3 -c "
from monitoring.feedback_system import FeedbackSystem
fs = FeedbackSystem()
status = fs.get_system_status()
print('System Health:', status['system_health'])
print('Active Models:', status['active_models'])
"
```

## 📚 Next Steps

### 1. Explore the System
- Try different dashboard tabs
- Test various API endpoints
- Generate insights with your own data

### 2. Customize for Your Business
- Modify `config.py` for your specific needs
- Add your own data sources
- Customize the dashboard layout

### 3. Integrate with Your Systems
- Connect to your databases
- Set up automated data pipelines
- Integrate with your existing tools

### 4. Scale the System
- Deploy to production servers
- Set up monitoring and alerting
- Implement backup and recovery

### 5. Advanced Usage
- Train custom models
- Add new insight types
- Implement real-time processing

## 📞 Support

If you encounter any issues or need help:

1. **Check the logs** for error messages
2. **Run the test suite** to identify problems
3. **Review the configuration** settings
4. **Check system requirements** and dependencies

## 🎉 Congratulations!

You've successfully set up and are running the SLM Business Insights system! The system is now ready to provide powerful business intelligence capabilities that will help drive revenue increases and cost reductions through advanced AI/ML models and real-time insights.

### Quick Reference

| Component | URL | Description |
|-----------|-----|-------------|
| API | http://localhost:5000 | REST API with interactive docs |
| Dashboard | http://localhost:8501 | Streamlit web interface |
| Health Check | http://localhost:5000/health | System status endpoint |

### Useful Commands

```bash
# Start everything
python3 app.py --mode all

# Run tests
python3 simple_test.py

# Check system status
curl http://localhost:5000/health

# View logs
tail -f logs/system.log
```

Happy analyzing! 🚀📊💡
