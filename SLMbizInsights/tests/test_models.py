"""
Tests for model components
"""
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hierarchical_model import (
    HighLevelModule, LowLevelModule, HierarchicalModel, 
    MultiTaskHierarchicalModel, create_hierarchical_model
)
from models.insight_models import (
    CustomerSegmentationModel, SentimentAnalysisModel, 
    SalesForecastingModel, PriceOptimizationModel, ChurnPredictionModel
)
from models.recommendation_engine import (
    MarketingRecommendationEngine, PricingRecommendationEngine,
    SupplyChainRecommendationEngine, RecommendationEngine
)

class TestHierarchicalModel:
    """Test hierarchical model components"""
    
    def test_high_level_module_init(self):
        """Test HighLevelModule initialization"""
        module = HighLevelModule(input_size=768, hidden_size=768, num_layers=4)
        assert module.bert is not None
        assert module.attention is not None
        assert module.strategic_classifier is not None
    
    def test_low_level_module_init(self):
        """Test LowLevelModule initialization"""
        module = LowLevelModule(input_size=512, hidden_size=512)
        assert module.distilbert is not None
        assert module.feature_extractor is not None
        assert module.sentiment_head is not None
    
    def test_hierarchical_model_init(self):
        """Test HierarchicalModel initialization"""
        model = HierarchicalModel(
            vocab_size=1000,
            embedding_size=256,
            high_level_hidden_size=256,
            low_level_hidden_size=128,
            output_size=64
        )
        assert model.embedding is not None
        assert model.high_level_module is not None
        assert model.low_level_module is not None
        assert model.integration_layer is not None
    
    def test_hierarchical_model_forward(self):
        """Test HierarchicalModel forward pass"""
        model = HierarchicalModel(
            vocab_size=1000,
            embedding_size=256,
            high_level_hidden_size=256,
            low_level_hidden_size=128,
            output_size=64
        )
        
        # Create sample input
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Check outputs
        assert 'revenue_prediction' in outputs
        assert 'cost_optimization' in outputs
        assert 'customer_segmentation' in outputs
        assert 'risk_assessment' in outputs
        assert 'opportunity_ranking' in outputs
        
        # Check output shapes
        assert outputs['revenue_prediction'].shape == (batch_size, 1)
        assert outputs['customer_segmentation'].shape == (batch_size, 10)
    
    def test_multi_task_model_init(self):
        """Test MultiTaskHierarchicalModel initialization"""
        model = MultiTaskHierarchicalModel(
            vocab_size=1000,
            embedding_size=256,
            high_level_hidden_size=256,
            low_level_hidden_size=128
        )
        assert model.base_model is not None
        assert len(model.insight_heads) > 0
        assert 'customer_lifetime_value' in model.insight_heads
    
    def test_create_hierarchical_model(self):
        """Test model factory function"""
        # Test base model creation
        base_model = create_hierarchical_model('base', vocab_size=1000)
        assert isinstance(base_model, HierarchicalModel)
        
        # Test multi-task model creation
        multi_task_model = create_hierarchical_model('multi_task', vocab_size=1000)
        assert isinstance(multi_task_model, MultiTaskHierarchicalModel)

class TestInsightModels:
    """Test insight model components"""
    
    def test_customer_segmentation_model(self):
        """Test CustomerSegmentationModel"""
        model = CustomerSegmentationModel(n_clusters=3)
        
        # Create test data
        test_data = pd.DataFrame({
            'age': [25, 35, 45, 55, 65],
            'income': [30000, 50000, 70000, 90000, 110000],
            'spending': [1000, 2000, 3000, 4000, 5000]
        })
        
        # Fit model
        model.fit(test_data, ['age', 'income', 'spending'])
        
        # Test prediction
        predictions = model.predict(test_data)
        assert len(predictions) == len(test_data)
        assert len(set(predictions)) <= 3  # Should have at most 3 clusters
        
        # Test cluster analysis
        analysis = model.analyze_clusters(test_data)
        assert len(analysis) == 3
        assert 'cluster_0' in analysis
    
    def test_sentiment_analysis_model(self):
        """Test SentimentAnalysisModel"""
        model = SentimentAnalysisModel()
        
        # Test data preparation
        texts = ["Great product!", "Not bad", "Terrible service"]
        labels = [2, 1, 0]  # Positive, Neutral, Negative
        
        # Test data preparation
        dataset = model.prepare_data(texts, labels)
        assert len(dataset) == 3
        assert 'input_ids' in dataset.columns
        assert 'attention_mask' in dataset.columns
    
    def test_sales_forecasting_model(self):
        """Test SalesForecastingModel"""
        model = SalesForecastingModel(method='arima')
        
        # Create test time series data
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        sales_data = pd.DataFrame({
            'date': dates,
            'sales': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
        })
        
        # Fit model
        model.fit(sales_data, 'date', 'sales')
        
        # Test prediction
        forecast = model.predict(periods=3)
        assert len(forecast) == 3
        assert 'date' in forecast.columns
        assert 'forecast' in forecast.columns
    
    def test_price_optimization_model(self):
        """Test PriceOptimizationModel"""
        model = PriceOptimizationModel()
        
        # Create test price data
        price_data = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_001', 'PROD_002', 'PROD_002'],
            'price': [100, 110, 200, 190],
            'quantity': [50, 45, 30, 35]
        })
        
        # Fit model
        model.fit(price_data)
        
        # Test price optimization
        optimal_price = model.optimize_price('PROD_001', cost=60)
        assert optimal_price > 60  # Should be higher than cost
        
        # Test batch optimization
        product_costs = {'PROD_001': 60, 'PROD_002': 120}
        optimal_prices = model.optimize_all_prices(product_costs)
        assert len(optimal_prices) == 2
        assert 'PROD_001' in optimal_prices
        assert 'PROD_002' in optimal_prices
    
    def test_churn_prediction_model(self):
        """Test ChurnPredictionModel"""
        model = ChurnPredictionModel()
        
        # Create test customer data
        customer_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'last_purchase_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'purchase_count': [5, 10, 2],
            'tenure_days': [365, 730, 90],
            'total_spend': [1000, 2000, 200],
            'churned': [0, 0, 1]
        })
        
        # Fit model
        model.fit(customer_data, 'churned')
        
        # Test prediction
        predictions = model.predict(customer_data)
        assert len(predictions) == len(customer_data)
        assert all(0 <= p <= 1 for p in predictions)  # Probabilities should be between 0 and 1

class TestRecommendationEngines:
    """Test recommendation engine components"""
    
    def test_marketing_recommendation_engine(self):
        """Test MarketingRecommendationEngine"""
        engine = MarketingRecommendationEngine()
        
        # Create test customer data
        customer_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004', 'CUST_005'],
            'total_spend': [1000, 2000, 500, 3000, 1500],
            'purchase_count': [5, 10, 2, 15, 7],
            'days_since_last_purchase': [10, 5, 90, 2, 30],
            'age': [25, 35, 45, 28, 40],
            'segment': ['Premium', 'Standard', 'Basic', 'Premium', 'Standard']
        })
        
        segments = np.array([0, 1, 2, 0, 1])
        
        # Test segment analysis
        analysis = engine.analyze_customer_segments(customer_data, segments)
        assert len(analysis) == 3
        assert 'segment_0' in analysis
        
        # Test marketing recommendations
        recommendations = engine.generate_marketing_recommendations(customer_data, segments)
        assert len(recommendations) == 3
        assert 'segment_0' in recommendations
        
        # Test ROI calculation
        roi_estimates = engine.calculate_marketing_roi(recommendations, customer_data, segments)
        assert len(roi_estimates) == 3
    
    def test_pricing_recommendation_engine(self):
        """Test PricingRecommendationEngine"""
        engine = PricingRecommendationEngine()
        
        # Create test sales data
        sales_data = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_001', 'PROD_002', 'PROD_002'],
            'price': [100, 110, 200, 190],
            'quantity': [50, 45, 30, 35]
        })
        
        # Test elasticity calculation
        elasticity = engine.calculate_price_elasticity(sales_data)
        assert len(elasticity) > 0
        
        # Test price optimization
        products = ['PROD_001', 'PROD_002']
        current_prices = {'PROD_001': 100, 'PROD_002': 200}
        costs = {'PROD_001': 60, 'PROD_002': 120}
        demand_forecasts = {'PROD_001': 50, 'PROD_002': 30}
        
        optimization_results = engine.optimize_pricing_strategy(
            products, current_prices, costs, demand_forecasts
        )
        assert len(optimization_results) == 2
        assert 'PROD_001' in optimization_results
        assert 'PROD_002' in optimization_results
    
    def test_supply_chain_recommendation_engine(self):
        """Test SupplyChainRecommendationEngine"""
        engine = SupplyChainRecommendationEngine()
        
        # Create test supplier data
        supplier_data = pd.DataFrame({
            'supplier_id': ['SUP_001', 'SUP_002', 'SUP_003'],
            'on_time': [True, False, True],
            'quality_rating': [4.5, 3.0, 4.8],
            'cost': [100, 80, 120],
            'lead_time': [7, 14, 5]
        })
        
        # Test supplier performance analysis
        analysis = engine.analyze_supplier_performance(supplier_data)
        assert len(analysis) == 3
        assert 'SUP_001' in analysis
        
        # Test supplier recommendations
        recommendations = engine.generate_supplier_recommendations(analysis)
        assert len(recommendations) == 3
        
        # Test inventory optimization
        demand_data = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_002'],
            'date': ['2023-01-01', '2023-01-02'],
            'demand': [10, 15]
        })
        
        inventory_data = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_002'],
            'current_stock': [50, 30]
        })
        
        inventory_recs = engine.optimize_inventory_levels(demand_data, inventory_data, pd.DataFrame())
        assert len(inventory_recs) == 2
    
    def test_recommendation_engine(self):
        """Test main RecommendationEngine"""
        engine = RecommendationEngine()
        
        # Create test data
        customer_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'total_spend': [1000, 2000, 500],
            'purchase_count': [5, 10, 2],
            'age': [25, 35, 45]
        })
        
        sales_data = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_002'],
            'amount': [100, 200],
            'quantity': [10, 5]
        })
        
        # Test comprehensive recommendations
        recommendations = engine.generate_comprehensive_recommendations(
            customer_data, sales_data, pd.DataFrame(), pd.DataFrame()
        )
        
        assert 'marketing' in recommendations
        assert 'pricing' in recommendations
        assert 'supply_chain' in recommendations
        assert 'summary' in recommendations

if __name__ == "__main__":
    pytest.main([__file__])
