"""
Pytest configuration and fixtures
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta

@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    return pd.DataFrame({
        'customer_id': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004', 'CUST_005'],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com'],
        'age': [25, 35, 45, 28, 40],
        'total_spend': [1000, 2000, 500, 3000, 1500],
        'purchase_count': [5, 10, 2, 15, 7],
        'retention_rate': [0.8, 0.9, 0.6, 0.95, 0.7],
        'last_purchase_date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-01-15', '2023-02-15'],
        'location': ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ']
    })

@pytest.fixture
def sample_sales_data():
    """Sample sales data for testing"""
    return pd.DataFrame({
        'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003', 'TXN_004', 'TXN_005'],
        'customer_id': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004', 'CUST_005'],
        'product_id': ['PROD_001', 'PROD_002', 'PROD_001', 'PROD_003', 'PROD_002'],
        'amount': [100, 150, 200, 75, 300],
        'quantity': [2, 1, 3, 1, 2],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'region': ['North', 'South', 'East', 'West', 'North']
    })

@pytest.fixture
def sample_feedback_data():
    """Sample feedback data for testing"""
    return pd.DataFrame({
        'feedback_id': ['FB_001', 'FB_002', 'FB_003', 'FB_004', 'FB_005'],
        'customer_id': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004', 'CUST_005'],
        'product_id': ['PROD_001', 'PROD_002', 'PROD_001', 'PROD_003', 'PROD_002'],
        'rating': [5, 4, 2, 5, 3],
        'text': [
            'Great product, very satisfied!',
            'Good quality, would recommend',
            'Poor quality, disappointed',
            'Excellent service and product',
            'Average product, nothing special'
        ],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'category': ['Product', 'Service', 'Product', 'Service', 'Product']
    })

@pytest.fixture
def sample_inventory_data():
    """Sample inventory data for testing"""
    return pd.DataFrame({
        'product_id': ['PROD_001', 'PROD_002', 'PROD_003', 'PROD_004', 'PROD_005'],
        'current_stock': [100, 50, 200, 75, 150],
        'reorder_point': [50, 25, 100, 40, 75],
        'supplier_id': ['SUP_001', 'SUP_002', 'SUP_001', 'SUP_003', 'SUP_002'],
        'cost': [25, 40, 15, 60, 35]
    })

@pytest.fixture
def sample_supplier_data():
    """Sample supplier data for testing"""
    return pd.DataFrame({
        'supplier_id': ['SUP_001', 'SUP_002', 'SUP_003', 'SUP_004', 'SUP_005'],
        'name': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E'],
        'on_time': [True, False, True, True, False],
        'quality_rating': [4.5, 3.0, 4.8, 4.2, 3.5],
        'cost': [100, 80, 120, 90, 110],
        'lead_time': [7, 14, 5, 10, 12]
    })

@pytest.fixture
def temp_db_file():
    """Temporary database file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)

@pytest.fixture
def temp_csv_file():
    """Temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)

@pytest.fixture
def sample_time_series_data():
    """Sample time series data for testing"""
    dates = pd.date_range('2023-01-01', periods=12, freq='M')
    return pd.DataFrame({
        'date': dates,
        'sales': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100],
        'demand': [950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050]
    })

@pytest.fixture
def sample_price_data():
    """Sample price data for testing"""
    return pd.DataFrame({
        'product_id': ['PROD_001', 'PROD_001', 'PROD_002', 'PROD_002', 'PROD_003', 'PROD_003'],
        'price': [100, 110, 200, 190, 150, 160],
        'quantity': [50, 45, 30, 35, 40, 38],
        'cost': [60, 60, 120, 120, 90, 90]
    })

@pytest.fixture
def sample_text_data():
    """Sample text data for testing"""
    return [
        "Great product, very satisfied with the quality!",
        "Poor customer service, took too long to respond",
        "Excellent value for money, would recommend",
        "Product arrived damaged, need better packaging",
        "Fast delivery and good quality product",
        "Not as described, disappointed with purchase",
        "Amazing experience, will buy again",
        "Average product, nothing special",
        "Outstanding customer support team",
        "Product quality could be better"
    ]

@pytest.fixture
def sample_sentiment_labels():
    """Sample sentiment labels for testing"""
    return [2, 0, 2, 0, 2, 0, 2, 1, 2, 0]  # 2=Positive, 1=Neutral, 0=Negative

@pytest.fixture
def mock_api_response():
    """Mock API response for testing"""
    return {
        'status': 'success',
        'data': {
            'customers': [
                {'id': 'CUST_001', 'name': 'John Doe', 'email': 'john@example.com'},
                {'id': 'CUST_002', 'name': 'Jane Smith', 'email': 'jane@example.com'}
            ],
            'sales': [
                {'id': 'SALE_001', 'amount': 100, 'date': '2023-01-01'},
                {'id': 'SALE_002', 'amount': 200, 'date': '2023-01-02'}
            ]
        }
    }

@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing"""
    return {
        'vocab_size': 1000,
        'embedding_size': 256,
        'high_level_hidden_size': 256,
        'low_level_hidden_size': 128,
        'output_size': 64,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10
    }

@pytest.fixture
def sample_metrics():
    """Sample performance metrics for testing"""
    return {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85,
        'mse': 0.15,
        'r2_score': 0.78
    }

@pytest.fixture
def sample_feedback_entry():
    """Sample feedback entry for testing"""
    return {
        'user_id': 'user1',
        'insight_id': 'insight1',
        'rating': 5,
        'feedback_text': 'Great insights!',
        'timestamp': datetime.now(),
        'model_version': '1.0',
        'prediction_confidence': 0.95
    }

@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory"""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add slow marker to tests that take more than 1 second
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid or "test_api" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to unit tests
        if "test_" in item.nodeid and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
