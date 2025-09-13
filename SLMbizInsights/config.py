"""
Configuration settings for SLM Business Insights
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/slm_bi')
    MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/slm_bi')
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # External API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    SALESFORCE_CLIENT_ID = os.getenv('SALESFORCE_CLIENT_ID', '')
    SALESFORCE_CLIENT_SECRET = os.getenv('SALESFORCE_CLIENT_SECRET', '')
    
    # Model Configuration
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', './models/cache')
    MAX_SEQUENCE_LENGTH = int(os.getenv('MAX_SEQUENCE_LENGTH', 512))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    
    # Security
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', 'dev-encryption-key')
    JWT_SECRET = os.getenv('JWT_SECRET', 'dev-jwt-secret')
    
    # Monitoring
    PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', 8000))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Data Sources
    CRM_API_URL = os.getenv('CRM_API_URL', 'https://api.crm.example.com')
    SALES_API_URL = os.getenv('SALES_API_URL', 'https://api.sales.example.com')
    FINANCIAL_API_URL = os.getenv('FINANCIAL_API_URL', 'https://api.financial.example.com')
    
    # File Paths
    DATA_DIR = './data'
    RAW_DATA_DIR = './data/raw'
    PROCESSED_DATA_DIR = './data/processed'
    EXTERNAL_DATA_DIR = './data/external'
    MODELS_DIR = './models'
    TRAINED_MODELS_DIR = './models/trained'
    SAVED_MODELS_DIR = './models/saved'
