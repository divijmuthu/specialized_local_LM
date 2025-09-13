"""
ETL Pipeline for data processing and transformation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.filesystem import FileSensor
import os
from config import Config

logger = logging.getLogger(__name__)

class ETLPipeline:
    """Main ETL pipeline class"""
    
    def __init__(self):
        self.raw_data_dir = Config.RAW_DATA_DIR
        self.processed_data_dir = Config.PROCESSED_DATA_DIR
        self.external_data_dir = Config.EXTERNAL_DATA_DIR
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.external_data_dir, exist_ok=True)
    
    def extract_data(self) -> Dict[str, pd.DataFrame]:
        """Extract data from various sources"""
        logger.info("Starting data extraction")
        
        data = {}
        
        # Load data from CSV files
        csv_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.csv')]
        
        for file in csv_files:
            filepath = os.path.join(self.raw_data_dir, file)
            name = file.replace('.csv', '')
            try:
                df = pd.read_csv(filepath)
                data[name] = df
                logger.info(f"Loaded {name}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        return data
    
    def transform_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Transform and clean data"""
        logger.info("Starting data transformation")
        
        transformed_data = {}
        
        for name, df in data.items():
            try:
                # Basic data cleaning
                df_clean = self._clean_dataframe(df)
                
                # Apply specific transformations based on data type
                if name == 'customers':
                    df_clean = self._transform_customer_data(df_clean)
                elif name == 'sales':
                    df_clean = self._transform_sales_data(df_clean)
                elif name == 'feedback':
                    df_clean = self._transform_feedback_data(df_clean)
                elif name == 'purchases':
                    df_clean = self._transform_purchase_data(df_clean)
                
                transformed_data[name] = df_clean
                logger.info(f"Transformed {name}: {len(df_clean)} records")
                
            except Exception as e:
                logger.error(f"Error transforming {name}: {e}")
                transformed_data[name] = df
        
        return transformed_data
    
    def load_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Load processed data to storage"""
        logger.info("Starting data loading")
        
        for name, df in data.items():
            try:
                # Save to processed data directory
                filepath = os.path.join(self.processed_data_dir, f"{name}_processed.csv")
                df.to_csv(filepath, index=False)
                logger.info(f"Saved processed {name} to {filepath}")
                
                # Also save as parquet for better performance
                parquet_path = os.path.join(self.processed_data_dir, f"{name}_processed.parquet")
                df.to_parquet(parquet_path, index=False)
                logger.info(f"Saved processed {name} to {parquet_path}")
                
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic dataframe cleaning"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        # For numeric columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col].fillna(mode_value[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        return df
    
    def _transform_customer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform customer data"""
        # Calculate customer lifetime value
        if 'total_spend' in df.columns and 'retention_rate' in df.columns:
            df['CLV'] = df['total_spend'] / (1 - df['retention_rate'])
        
        # Create age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 45, 55, 100], 
                                   labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Create customer tenure
        if 'registration_date' in df.columns:
            df['registration_date'] = pd.to_datetime(df['registration_date'])
            df['tenure_days'] = (datetime.now() - df['registration_date']).dt.days
            df['tenure_months'] = df['tenure_days'] / 30
        
        return df
    
    def _transform_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform sales data"""
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_week'] = df['date'].dt.day_name()
        
        # Calculate total revenue per transaction
        if 'amount' in df.columns and 'quantity' in df.columns:
            df['total_revenue'] = df['amount'] * df['quantity']
        
        # Create price tiers
        if 'amount' in df.columns:
            df['price_tier'] = pd.cut(df['amount'], 
                                    bins=[0, 50, 100, 200, 1000], 
                                    labels=['Low', 'Medium', 'High', 'Premium'])
        
        return df
    
    def _transform_feedback_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform feedback data"""
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Create sentiment categories based on rating
        if 'rating' in df.columns:
            df['sentiment'] = df['rating'].apply(
                lambda x: 'Positive' if x >= 4 else 'Negative' if x <= 2 else 'Neutral'
            )
        
        # Extract text length
        if 'text' in df.columns:
            df['text_length'] = df['text'].str.len()
            df['word_count'] = df['text'].str.split().str.len()
        
        return df
    
    def _transform_purchase_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform purchase data"""
        # Convert date column
        if 'purchase_date' in df.columns:
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        
        # Calculate purchase frequency
        if 'customer_id' in df.columns and 'purchase_date' in df.columns:
            customer_purchase_counts = df.groupby('customer_id').size()
            df['purchase_frequency'] = df['customer_id'].map(customer_purchase_counts)
        
        return df
    
    def run_pipeline(self) -> None:
        """Run the complete ETL pipeline"""
        logger.info("Starting ETL pipeline")
        
        try:
            # Extract
            raw_data = self.extract_data()
            
            # Transform
            processed_data = self.transform_data(raw_data)
            
            # Load
            self.load_data(processed_data)
            
            logger.info("ETL pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            raise

# Airflow DAG definition
default_args = {
    'owner': 'slm-bi',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'slm_bi_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for SLM Business Insights',
    schedule_interval='@daily',
    catchup=False,
    tags=['etl', 'business-insights']
)

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=lambda: ETLPipeline().extract_data(),
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=lambda: ETLPipeline().transform_data(ETLPipeline().extract_data()),
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=lambda: ETLPipeline().load_data(
        ETLPipeline().transform_data(ETLPipeline().extract_data())
    ),
    dag=dag
)

# Set task dependencies
extract_task >> transform_task >> load_task

# Data quality checks
def data_quality_check():
    """Perform data quality checks"""
    pipeline = ETLPipeline()
    processed_files = [f for f in os.listdir(pipeline.processed_data_dir) if f.endswith('.csv')]
    
    quality_report = {}
    
    for file in processed_files:
        filepath = os.path.join(pipeline.processed_data_dir, file)
        df = pd.read_csv(filepath)
        
        quality_report[file] = {
            'row_count': len(df),
            'null_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            'duplicate_count': df.duplicated().sum(),
            'columns': list(df.columns)
        }
    
    # Save quality report
    report_path = os.path.join(pipeline.processed_data_dir, 'quality_report.json')
    import json
    with open(report_path, 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    logger.info(f"Data quality report saved to {report_path}")
    return quality_report

quality_check_task = PythonOperator(
    task_id='data_quality_check',
    python_callable=data_quality_check,
    dag=dag
)

# Add quality check after load
load_task >> quality_check_task

# Data validation
def validate_data():
    """Validate processed data"""
    pipeline = ETLPipeline()
    
    # Check if required files exist
    required_files = ['customers_processed.csv', 'sales_processed.csv', 'feedback_processed.csv']
    
    for file in required_files:
        filepath = os.path.join(pipeline.processed_data_dir, file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file {file} not found")
        
        # Check if file has data
        df = pd.read_csv(filepath)
        if len(df) == 0:
            raise ValueError(f"File {file} is empty")
    
    logger.info("Data validation passed")
    return True

validation_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

# Add validation after quality check
quality_check_task >> validation_task

if __name__ == "__main__":
    # Run pipeline directly for testing
    pipeline = ETLPipeline()
    pipeline.run_pipeline()
