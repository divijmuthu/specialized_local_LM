"""
Tests for data pipeline components
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_collectors import (
    DataCollector, CRMDataCollector, SalesDataCollector, 
    FinancialDataCollector, SampleDataGenerator
)
from pipeline.etl_pipeline import ETLPipeline
from pipeline.data_preprocessing import (
    TextPreprocessor, EmbeddingGenerator, FeatureEngineer, DataPreprocessor
)

class TestDataCollectors:
    """Test data collection components"""
    
    def test_data_collector_init(self):
        """Test DataCollector initialization"""
        collector = DataCollector("https://api.example.com")
        assert collector.api_url == "https://api.example.com"
        assert collector.headers == {}
    
    def test_sample_data_generator(self):
        """Test sample data generation"""
        generator = SampleDataGenerator()
        
        # Test customer data generation
        customers = generator.generate_customer_data(10)
        assert len(customers) == 10
        assert 'customer_id' in customers.columns
        assert 'name' in customers.columns
        assert 'email' in customers.columns
        
        # Test sales data generation
        sales = generator.generate_sales_data(20)
        assert len(sales) == 20
        assert 'transaction_id' in sales.columns
        assert 'amount' in sales.columns
        assert 'quantity' in sales.columns
        
        # Test feedback data generation
        feedback = generator.generate_feedback_data(15)
        assert len(feedback) == 15
        assert 'feedback_id' in feedback.columns
        assert 'rating' in feedback.columns
        assert 'text' in feedback.columns
    
    @patch('requests.Session.get')
    def test_fetch_data_success(self, mock_get):
        """Test successful data fetching"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_get.return_value = mock_response
        
        collector = DataCollector("https://api.example.com")
        result = collector.fetch_data("test")
        
        assert result == {'data': 'test'}
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_fetch_data_failure(self, mock_get):
        """Test data fetching failure"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        collector = DataCollector("https://api.example.com")
        result = collector.fetch_data("test")
        
        assert result is None

class TestETLPipeline:
    """Test ETL pipeline components"""
    
    def test_etl_pipeline_init(self):
        """Test ETL pipeline initialization"""
        pipeline = ETLPipeline()
        assert pipeline.raw_data_dir is not None
        assert pipeline.processed_data_dir is not None
        assert pipeline.external_data_dir is not None
    
    def test_clean_dataframe(self):
        """Test dataframe cleaning"""
        pipeline = ETLPipeline()
        
        # Create test data with duplicates and missing values
        test_data = pd.DataFrame({
            'id': [1, 2, 2, 3, 4],
            'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
            'age': [25, 30, 30, 35, 40],
            'score': [85, 90, 90, None, 95]
        })
        
        cleaned_data = pipeline._clean_dataframe(test_data)
        
        # Check duplicates removed
        assert len(cleaned_data) == 4
        
        # Check missing values handled
        assert cleaned_data['name'].isnull().sum() == 0
        assert cleaned_data['score'].isnull().sum() == 0
    
    def test_transform_customer_data(self):
        """Test customer data transformation"""
        pipeline = ETLPipeline()
        
        test_data = pd.DataFrame({
            'total_spend': [1000, 2000, 1500],
            'retention_rate': [0.8, 0.9, 0.7],
            'age': [25, 35, 45],
            'registration_date': ['2022-01-01', '2021-06-15', '2023-03-10']
        })
        
        transformed_data = pipeline._transform_customer_data(test_data)
        
        # Check CLV calculation
        assert 'CLV' in transformed_data.columns
        assert transformed_data['CLV'].iloc[0] == 1000 / (1 - 0.8)
        
        # Check age groups
        assert 'age_group' in transformed_data.columns
        
        # Check tenure calculation
        assert 'tenure_days' in transformed_data.columns
        assert 'tenure_months' in transformed_data.columns
    
    def test_transform_sales_data(self):
        """Test sales data transformation"""
        pipeline = ETLPipeline()
        
        test_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'amount': [100, 150, 200],
            'quantity': [2, 3, 1]
        })
        
        transformed_data = pipeline._transform_sales_data(test_data)
        
        # Check date features
        assert 'year' in transformed_data.columns
        assert 'month' in transformed_data.columns
        assert 'quarter' in transformed_data.columns
        
        # Check revenue calculation
        assert 'total_revenue' in transformed_data.columns
        assert transformed_data['total_revenue'].iloc[0] == 200
        
        # Check price tiers
        assert 'price_tier' in transformed_data.columns

class TestDataPreprocessing:
    """Test data preprocessing components"""
    
    def test_text_preprocessor_init(self):
        """Test TextPreprocessor initialization"""
        preprocessor = TextPreprocessor()
        assert preprocessor.stop_words is not None
        assert preprocessor.lemmatizer is not None
    
    def test_clean_text(self):
        """Test text cleaning"""
        preprocessor = TextPreprocessor()
        
        test_text = "Hello! This is a test email: test@example.com. Visit https://example.com"
        cleaned = preprocessor.clean_text(test_text)
        
        assert "@" not in cleaned
        assert "http" not in cleaned
        assert "!" not in cleaned
        assert "hello" in cleaned.lower()
    
    def test_preprocess_text(self):
        """Test complete text preprocessing"""
        preprocessor = TextPreprocessor()
        
        test_text = "This is a great product! I love it very much."
        processed = preprocessor.preprocess_text(test_text)
        
        assert isinstance(processed, str)
        assert len(processed) > 0
        # Should remove stopwords and punctuation
        assert "is" not in processed  # stopword
        assert "!" not in processed  # punctuation
    
    def test_extract_features(self):
        """Test text feature extraction"""
        preprocessor = TextPreprocessor()
        
        test_text = "This is a great product!"
        features = preprocessor.extract_features(test_text)
        
        assert 'text_length' in features
        assert 'word_count' in features
        assert 'sentence_count' in features
        assert 'positive_word_ratio' in features
        assert features['text_length'] > 0
        assert features['word_count'] > 0
    
    def test_feature_engineer_init(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        assert engineer.scalers == {}
        assert engineer.encoders == {}
        assert engineer.vectorizers == {}
    
    def test_create_customer_features(self):
        """Test customer feature creation"""
        engineer = FeatureEngineer()
        
        test_data = pd.DataFrame({
            'total_spend': [1000, 2000, 1500],
            'retention_rate': [0.8, 0.9, 0.7],
            'last_purchase_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'purchase_count': [5, 10, 7]
        })
        
        features = engineer.create_customer_features(test_data)
        
        # Check CLV calculation
        assert 'CLV' in features.columns
        
        # Check RFM features
        assert 'recency' in features.columns
        assert 'frequency' in features.columns
        assert 'monetary' in features.columns
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        engineer = FeatureEngineer()
        
        test_data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B'],
            'status': ['active', 'inactive', 'active', 'pending', 'active']
        })
        
        encoded_data = engineer.encode_categorical_features(
            test_data, ['category', 'status']
        )
        
        # Check that categorical columns are encoded
        assert encoded_data['category'].dtype in ['int64', 'int32']
        assert encoded_data['status'].dtype in ['int64', 'int32']
        
        # Check that encoders are stored
        assert 'category' in engineer.encoders
        assert 'status' in engineer.encoders
    
    def test_scale_numerical_features(self):
        """Test numerical feature scaling"""
        engineer = FeatureEngineer()
        
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        scaled_data = engineer.scale_numerical_features(
            test_data, ['feature1', 'feature2']
        )
        
        # Check that features are scaled
        assert scaled_data['feature1'].mean() < 0.1  # Should be close to 0
        assert abs(scaled_data['feature1'].std() - 1.0) < 0.1  # Should be close to 1
        
        # Check that scalers are stored
        assert 'feature1' in engineer.scalers
        assert 'feature2' in engineer.scalers

class TestDataPreprocessor:
    """Test main data preprocessor"""
    
    def test_data_preprocessor_init(self):
        """Test DataPreprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.text_preprocessor is not None
        assert preprocessor.embedding_generator is not None
        assert preprocessor.feature_engineer is not None
    
    def test_preprocess_data(self):
        """Test data preprocessing pipeline"""
        preprocessor = DataPreprocessor()
        
        # Create test data
        test_data = {
            'customers': pd.DataFrame({
                'customer_id': ['CUST_001', 'CUST_002'],
                'total_spend': [1000, 2000],
                'retention_rate': [0.8, 0.9]
            }),
            'feedback': pd.DataFrame({
                'feedback_id': ['FB_001', 'FB_002'],
                'text': ['Great product!', 'Not bad.'],
                'rating': [5, 3]
            })
        }
        
        processed_data = preprocessor.preprocess_data(test_data)
        
        # Check that data is processed
        assert 'customers' in processed_data
        assert 'feedback' in processed_data
        
        # Check customer features
        customers = processed_data['customers']
        assert 'CLV' in customers.columns
        
        # Check feedback features
        feedback = processed_data['feedback']
        assert 'cleaned_text' in feedback.columns
        assert 'sentiment' in feedback.columns

if __name__ == "__main__":
    pytest.main([__file__])
