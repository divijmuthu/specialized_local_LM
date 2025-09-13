"""
Tests for API endpoints
"""
import pytest
import json
import requests
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app

class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_customer_segmentation(self, client):
        """Test customer segmentation endpoint"""
        test_data = {
            'customers': [
                {'customer_id': 'CUST_001', 'age': 25, 'income': 30000, 'spending': 1000},
                {'customer_id': 'CUST_002', 'age': 35, 'income': 50000, 'spending': 2000},
                {'customer_id': 'CUST_003', 'age': 45, 'income': 70000, 'spending': 3000}
            ],
            'n_clusters': 3
        }
        
        response = client.post('/api/insights/customer-segmentation', 
                             json=test_data)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'segments' in data
        assert 'segment_analysis' in data
        assert len(data['segments']) == 3
    
    def test_customer_segmentation_no_data(self, client):
        """Test customer segmentation with no data"""
        response = client.post('/api/insights/customer-segmentation', 
                             json={})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_sentiment_analysis(self, client):
        """Test sentiment analysis endpoint"""
        test_data = {
            'texts': [
                'Great product, very satisfied!',
                'Poor quality, disappointed',
                'Average product, nothing special'
            ]
        }
        
        response = client.post('/api/insights/sentiment-analysis', 
                             json=test_data)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'sentiments' in data
        assert 'summary' in data
        assert len(data['sentiments']) == 3
        assert data['summary']['total_texts'] == 3
    
    def test_sentiment_analysis_no_data(self, client):
        """Test sentiment analysis with no data"""
        response = client.post('/api/insights/sentiment-analysis', 
                             json={})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_sales_forecasting(self, client):
        """Test sales forecasting endpoint"""
        test_data = {
            'sales_data': [
                {'date': '2023-01-01', 'sales': 1000},
                {'date': '2023-02-01', 'sales': 1100},
                {'date': '2023-03-01', 'sales': 1200}
            ],
            'periods': 3
        }
        
        response = client.post('/api/insights/sales-forecasting', 
                             json=test_data)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'forecast' in data
        assert 'historical_data' in data
        assert len(data['forecast']) == 3
    
    def test_sales_forecasting_no_data(self, client):
        """Test sales forecasting with no data"""
        response = client.post('/api/insights/sales-forecasting', 
                             json={})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_price_optimization(self, client):
        """Test price optimization endpoint"""
        test_data = {
            'price_data': [
                {'product_id': 'PROD_001', 'price': 100, 'quantity': 50},
                {'product_id': 'PROD_001', 'price': 110, 'quantity': 45},
                {'product_id': 'PROD_002', 'price': 200, 'quantity': 30}
            ]
        }
        
        response = client.post('/api/insights/price-optimization', 
                             json=test_data)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'optimization_results' in data
        assert 'summary' in data
        assert len(data['optimization_results']) > 0
    
    def test_churn_prediction(self, client):
        """Test churn prediction endpoint"""
        test_data = {
            'customer_data': [
                {
                    'customer_id': 'CUST_001',
                    'last_purchase_date': '2023-01-01',
                    'purchase_count': 5,
                    'tenure_days': 365,
                    'total_spend': 1000
                },
                {
                    'customer_id': 'CUST_002',
                    'last_purchase_date': '2023-02-01',
                    'purchase_count': 10,
                    'tenure_days': 730,
                    'total_spend': 2000
                }
            ]
        }
        
        response = client.post('/api/insights/churn-prediction', 
                             json=test_data)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'predictions' in data
        assert 'summary' in data
        assert len(data['predictions']) == 2
    
    def test_marketing_recommendations(self, client):
        """Test marketing recommendations endpoint"""
        test_data = {
            'customer_data': [
                {'customer_id': 'CUST_001', 'age': 25, 'total_spend': 1000, 'purchase_count': 5},
                {'customer_id': 'CUST_002', 'age': 35, 'total_spend': 2000, 'purchase_count': 10},
                {'customer_id': 'CUST_003', 'age': 45, 'total_spend': 500, 'purchase_count': 2}
            ],
            'n_clusters': 3
        }
        
        response = client.post('/api/recommendations/marketing', 
                             json=test_data)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'segments' in data
        assert 'marketing_recommendations' in data
        assert 'segment_analysis' in data
    
    def test_comprehensive_recommendations(self, client):
        """Test comprehensive recommendations endpoint"""
        test_data = {
            'customer_data': [
                {'customer_id': 'CUST_001', 'age': 25, 'total_spend': 1000},
                {'customer_id': 'CUST_002', 'age': 35, 'total_spend': 2000}
            ],
            'sales_data': [
                {'product_id': 'PROD_001', 'amount': 100, 'quantity': 10},
                {'product_id': 'PROD_002', 'amount': 200, 'quantity': 5}
            ]
        }
        
        response = client.post('/api/recommendations/comprehensive', 
                             json=test_data)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'marketing' in data
        assert 'pricing' in data
        assert 'supply_chain' in data
        assert 'summary' in data
    
    def test_sample_data_generation(self, client):
        """Test sample data generation endpoint"""
        response = client.get('/api/data/sample')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'customers' in data
        assert 'sales' in data
        assert 'feedback' in data
        assert len(data['customers']) > 0
        assert len(data['sales']) > 0
        assert len(data['feedback']) > 0
    
    def test_model_training(self, client):
        """Test model training endpoint"""
        test_data = {
            'data': {
                'customers': [
                    {'customer_id': 'CUST_001', 'age': 25, 'total_spend': 1000},
                    {'customer_id': 'CUST_002', 'age': 35, 'total_spend': 2000}
                ]
            }
        }
        
        response = client.post('/api/models/train', 
                             json=test_data)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'training_results' in data
        assert 'status' in data
        assert data['status'] == 'completed'
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/nonexistent-endpoint')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Endpoint not found'
    
    def test_invalid_json(self, client):
        """Test invalid JSON handling"""
        response = client.post('/api/insights/customer-segmentation',
                             data='invalid json',
                             content_type='application/json')
        assert response.status_code == 400

class TestAPIIntegration:
    """Test API integration scenarios"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_end_to_end_workflow(self, client):
        """Test complete end-to-end workflow"""
        # 1. Generate sample data
        response = client.get('/api/data/sample')
        assert response.status_code == 200
        sample_data = json.loads(response.data)
        
        # 2. Run customer segmentation
        segmentation_data = {
            'customers': sample_data['customers'][:20],
            'n_clusters': 3
        }
        response = client.post('/api/insights/customer-segmentation', 
                             json=segmentation_data)
        assert response.status_code == 200
        segmentation_result = json.loads(response.data)
        
        # 3. Generate marketing recommendations
        marketing_data = {
            'customer_data': sample_data['customers'][:20],
            'n_clusters': 3
        }
        response = client.post('/api/recommendations/marketing', 
                             json=marketing_data)
        assert response.status_code == 200
        marketing_result = json.loads(response.data)
        
        # 4. Run sentiment analysis
        sentiment_data = {
            'texts': [f['text'] for f in sample_data['feedback'][:5]]
        }
        response = client.post('/api/insights/sentiment-analysis', 
                             json=sentiment_data)
        assert response.status_code == 200
        sentiment_result = json.loads(response.data)
        
        # Verify all results are valid
        assert len(segmentation_result['segments']) == 20
        assert len(marketing_result['marketing_recommendations']) > 0
        assert len(sentiment_result['sentiments']) == 5
    
    def test_error_handling_workflow(self, client):
        """Test error handling in workflow"""
        # Test with invalid data
        invalid_data = {
            'customers': [
                {'invalid_field': 'value'}  # Missing required fields
            ]
        }
        
        response = client.post('/api/insights/customer-segmentation', 
                             json=invalid_data)
        assert response.status_code == 400
        
        # Test with empty data
        response = client.post('/api/insights/customer-segmentation', 
                             json={'customers': []})
        assert response.status_code == 400

if __name__ == "__main__":
    pytest.main([__file__])
