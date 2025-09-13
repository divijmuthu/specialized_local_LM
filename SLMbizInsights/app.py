"""
Main application entry point for SLM Business Insights
"""
import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from api.app import app as api_app
from ui.dashboard import BusinessInsightsDashboard
import streamlit as st

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('slm_bi.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_api_server():
    """Run the Flask API server"""
    logger.info("Starting SLM Business Insights API server...")
    api_app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        debug=False
    )

def run_dashboard():
    """Run the Streamlit dashboard"""
    logger.info("Starting SLM Business Insights Dashboard...")
    dashboard_path = os.path.join(os.path.dirname(__file__), 'ui', 'dashboard.py')
    
    import subprocess
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', dashboard_path,
        '--server.port', '8501',
        '--server.address', '0.0.0.0'
    ])

def run_data_pipeline():
    """Run the data pipeline"""
    logger.info("Running data pipeline...")
    
    from pipeline.data_collectors import SampleDataGenerator, DataCollectionOrchestrator
    from pipeline.etl_pipeline import ETLPipeline
    from pipeline.data_preprocessing import DataPreprocessor
    
    try:
        # Generate sample data
        generator = SampleDataGenerator()
        sample_data = {
            'customers': generator.generate_customer_data(1000),
            'sales': generator.generate_sales_data(2000),
            'feedback': generator.generate_feedback_data(1000)
        }
        
        # Run ETL pipeline
        etl_pipeline = ETLPipeline()
        etl_pipeline.run_pipeline()
        
        # Run data preprocessing
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_data(sample_data)
        preprocessor.save_processed_data(processed_data, Config.PROCESSED_DATA_DIR)
        
        logger.info("Data pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running data pipeline: {e}")
        raise

def run_model_training():
    """Run model training"""
    logger.info("Running model training...")
    
    from models.training_pipeline import ModelTrainer
    from pipeline.data_collectors import SampleDataGenerator
    from pipeline.data_preprocessing import DataPreprocessor
    
    try:
        # Generate and preprocess sample data
        generator = SampleDataGenerator()
        sample_data = {
            'customers': generator.generate_customer_data(1000),
            'sales': generator.generate_sales_data(2000),
            'feedback': generator.generate_feedback_data(1000)
        }
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_data(sample_data)
        
        # Train models
        trainer = ModelTrainer('multi_task', 'small')
        results = trainer.train_model(processed_data, num_epochs=2)
        
        logger.info("Model training completed successfully!")
        logger.info(f"Test metrics: {results['test_metrics']}")
        
    except Exception as e:
        logger.error(f"Error running model training: {e}")
        raise

def run_insights_generation():
    """Run insights generation"""
    logger.info("Generating business insights...")
    
    from models.insight_models import InsightModelsManager
    from pipeline.data_collectors import SampleDataGenerator
    
    try:
        # Generate sample data
        generator = SampleDataGenerator()
        sample_data = {
            'customers': generator.generate_customer_data(500),
            'sales': generator.generate_sales_data(1000),
            'feedback': generator.generate_feedback_data(500)
        }
        
        # Initialize models manager
        models_manager = InsightModelsManager()
        
        # Train and test models
        models_manager.train_model('customer_segmentation', sample_data)
        models_manager.train_model('sentiment_analysis', sample_data)
        
        # Generate insights
        customer_insights = models_manager.get_insights('customer_segmentation', sample_data)
        sentiment_insights = models_manager.get_insights('nlp_insights', sample_data)
        
        logger.info("Business insights generated successfully!")
        logger.info(f"Customer segments: {len(customer_insights)}")
        logger.info(f"Sentiment insights: {len(sentiment_insights)}")
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise

def run_recommendations():
    """Run recommendations generation"""
    logger.info("Generating business recommendations...")
    
    from models.recommendation_engine import RecommendationEngine
    from pipeline.data_collectors import SampleDataGenerator
    
    try:
        # Generate sample data
        generator = SampleDataGenerator()
        sample_data = {
            'customers': generator.generate_customer_data(500),
            'sales': generator.generate_sales_data(1000),
            'feedback': generator.generate_feedback_data(500)
        }
        
        # Create sample inventory and supplier data
        import pandas as pd
        import numpy as np
        
        inventory_data = pd.DataFrame({
            'product_id': [f'PROD_{i:03d}' for i in range(1, 101)],
            'current_stock': np.random.randint(10, 1000, 100),
            'reorder_point': np.random.randint(50, 500, 100)
        })
        
        supplier_data = pd.DataFrame({
            'supplier_id': [f'SUP_{i:03d}' for i in range(1, 21)],
            'on_time': np.random.choice([True, False], 20, p=[0.8, 0.2]),
            'quality_rating': np.random.uniform(2, 5, 20),
            'cost': np.random.uniform(10, 100, 20),
            'lead_time': np.random.randint(3, 21, 20)
        })
        
        # Generate recommendations
        engine = RecommendationEngine()
        recommendations = engine.generate_comprehensive_recommendations(
            sample_data['customers'],
            sample_data['sales'],
            inventory_data,
            supplier_data
        )
        
        # Save recommendations
        engine.save_recommendations(recommendations, './data/recommendations.json')
        
        logger.info("Business recommendations generated successfully!")
        logger.info(f"Summary recommendations: {len(recommendations['summary'])}")
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise

def run_monitoring():
    """Run monitoring and feedback system"""
    logger.info("Starting monitoring and feedback system...")
    
    from monitoring.feedback_system import FeedbackSystem
    
    try:
        # Initialize feedback system
        feedback_system = FeedbackSystem()
        
        # Simulate some feedback
        feedback_system.process_feedback('user1', 'insight1', 5, 'Great insights!')
        feedback_system.process_feedback('user2', 'insight1', 3, 'Could be better')
        feedback_system.process_feedback('user3', 'insight2', 4, 'Helpful information')
        
        # Get system status
        status = feedback_system.get_system_status()
        
        # Run continuous learning cycle
        feedback_system.run_continuous_learning_cycle()
        
        logger.info("Monitoring and feedback system running successfully!")
        logger.info(f"System health: {status['system_health']:.2f}")
        
    except Exception as e:
        logger.error(f"Error running monitoring system: {e}")
        raise

def main():
    """Main application entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SLM Business Insights Application")
    parser.add_argument('--mode', choices=['api', 'dashboard', 'pipeline', 'train', 'insights', 'recommendations', 'monitoring', 'all'], 
                       default='all', help='Application mode to run')
    parser.add_argument('--port', type=int, default=5000, help='API server port')
    parser.add_argument('--host', default='0.0.0.0', help='API server host')
    
    args = parser.parse_args()
    
    # Update config if needed
    if args.port != 5000:
        Config.API_PORT = args.port
    if args.host != '0.0.0.0':
        Config.API_HOST = args.host
    
    logger.info(f"Starting SLM Business Insights in {args.mode} mode...")
    logger.info(f"Configuration: Host={Config.API_HOST}, Port={Config.API_PORT}")
    
    try:
        if args.mode == 'api':
            run_api_server()
        elif args.mode == 'dashboard':
            run_dashboard()
        elif args.mode == 'pipeline':
            run_data_pipeline()
        elif args.mode == 'train':
            run_model_training()
        elif args.mode == 'insights':
            run_insights_generation()
        elif args.mode == 'recommendations':
            run_recommendations()
        elif args.mode == 'monitoring':
            run_monitoring()
        elif args.mode == 'all':
            # Run all components in sequence
            logger.info("Running complete SLM Business Insights pipeline...")
            
            # 1. Data pipeline
            run_data_pipeline()
            
            # 2. Model training
            run_model_training()
            
            # 3. Insights generation
            run_insights_generation()
            
            # 4. Recommendations
            run_recommendations()
            
            # 5. Start monitoring (in background)
            import threading
            monitoring_thread = threading.Thread(target=run_monitoring)
            monitoring_thread.daemon = True
            monitoring_thread.start()
            
            # 6. Start API server
            logger.info("Starting API server and dashboard...")
            logger.info("API available at: http://localhost:5000")
            logger.info("Dashboard available at: http://localhost:8501")
            
            # Start API server (this will block)
            run_api_server()
            
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
