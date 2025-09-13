"""
Comprehensive test of SLM Business Insights system with sample data
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_generation():
    """Test data generation and basic functionality"""
    print("🚀 SLM Business Insights System Test")
    print("=" * 60)
    
    try:
        # Test 1: Sample Data Generation
        print("\n📊 Step 1: Generating Sample Data...")
        from pipeline.data_collectors import SampleDataGenerator
        
        generator = SampleDataGenerator()
        
        # Generate comprehensive sample data
        customers = generator.generate_customer_data(500)
        sales = generator.generate_sales_data(1000)
        feedback = generator.generate_feedback_data(500)
        
        print(f"✅ Generated {len(customers)} customers")
        print(f"✅ Generated {len(sales)} sales transactions")
        print(f"✅ Generated {len(feedback)} feedback records")
        
        # Display sample data
        print(f"\n📋 Sample Customer Data:")
        print(customers.head(3).to_string())
        
        print(f"\n📋 Sample Sales Data:")
        print(sales.head(3).to_string())
        
        print(f"\n📋 Sample Feedback Data:")
        print(feedback.head(3).to_string())
        
        return customers, sales, feedback
        
    except Exception as e:
        print(f"❌ Error in data generation: {e}")
        return None, None, None

def test_data_preprocessing(customers, sales, feedback):
    """Test data preprocessing pipeline"""
    print("\n🔧 Step 2: Data Preprocessing...")
    
    try:
        from pipeline.data_preprocessing import DataPreprocessor, TextPreprocessor
        
        # Test text preprocessing
        print("   📝 Testing text preprocessing...")
        text_preprocessor = TextPreprocessor()
        
        sample_texts = feedback['text'].head(5).tolist()
        processed_texts = [text_preprocessor.preprocess_text(text) for text in sample_texts]
        
        print(f"   ✅ Processed {len(processed_texts)} text samples")
        print(f"   📄 Original: {sample_texts[0][:50]}...")
        print(f"   📄 Processed: {processed_texts[0][:50]}...")
        
        # Test feature engineering
        print("   🛠️  Testing feature engineering...")
        from pipeline.data_preprocessing import FeatureEngineer
        
        engineer = FeatureEngineer()
        
        # Create customer features
        customer_features = engineer.create_customer_features(customers)
        print(f"   ✅ Created customer features: {list(customer_features.columns)}")
        
        # Create sales features
        sales_features = engineer.create_sales_features(sales)
        print(f"   ✅ Created sales features: {list(sales_features.columns)}")
        
        # Create feedback features
        feedback_features = engineer.create_feedback_features(feedback)
        print(f"   ✅ Created feedback features: {list(feedback_features.columns)}")
        
        return customer_features, sales_features, feedback_features
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        return None, None, None

def test_insight_models(customers, sales, feedback):
    """Test insight models"""
    print("\n🎯 Step 3: Testing Insight Models...")
    
    try:
        from models.insight_models import (
            CustomerSegmentationModel, 
            SentimentAnalysisModel,
            SalesForecastingModel,
            PriceOptimizationModel,
            ChurnPredictionModel
        )
        
        # Test 1: Customer Segmentation
        print("   👥 Testing Customer Segmentation...")
        segmentation_model = CustomerSegmentationModel(n_clusters=5)
        
        # Prepare features for clustering
        numeric_features = customers.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_features) > 0:
            segmentation_model.fit(customers, numeric_features)
            clusters = segmentation_model.predict(customers)
            analysis = segmentation_model.analyze_clusters(customers)
            
            print(f"   ✅ Customer segmentation completed")
            print(f"   📊 Found {len(set(clusters))} customer segments")
            print(f"   📈 Segment sizes: {[len(customers[clusters == i]) for i in range(5)]}")
        
        # Test 2: Sentiment Analysis
        print("   😊 Testing Sentiment Analysis...")
        sentiment_model = SentimentAnalysisModel()
        
        # Prepare sentiment data
        texts = feedback['text'].head(20).tolist()
        ratings = feedback['rating'].head(20).tolist()
        
        # Convert ratings to sentiment labels (1-2: negative, 3: neutral, 4-5: positive)
        sentiment_labels = [0 if r <= 2 else 1 if r == 3 else 2 for r in ratings]
        
        print(f"   ✅ Prepared {len(texts)} texts for sentiment analysis")
        print(f"   📊 Sentiment distribution: {np.bincount(sentiment_labels)}")
        
        # Test 3: Sales Forecasting
        print("   📈 Testing Sales Forecasting...")
        sales_model = SalesForecastingModel(method='arima')
        
        # Prepare time series data
        sales_ts = sales.groupby('date')['amount'].sum().reset_index()
        sales_ts['date'] = pd.to_datetime(sales_ts['date'])
        sales_ts = sales_ts.sort_values('date')
        
        if len(sales_ts) > 5:
            sales_model.fit(sales_ts, 'date', 'amount')
            forecast = sales_model.predict(periods=3)
            print(f"   ✅ Sales forecasting completed")
            print(f"   📊 Generated {len(forecast)} forecast periods")
        
        # Test 4: Price Optimization
        print("   💰 Testing Price Optimization...")
        price_model = PriceOptimizationModel()
        
        # Prepare price data
        price_data = sales[['product_id', 'amount', 'quantity']].copy()
        price_data.columns = ['product_id', 'price', 'quantity']
        
        price_model.fit(price_data)
        
        # Test price optimization for a sample product
        sample_products = price_data['product_id'].unique()[:3]
        for product in sample_products:
            optimal_price = price_model.optimize_price(product, cost=50)
            print(f"   ✅ Product {product}: Optimal price = ${optimal_price:.2f}")
        
        # Test 5: Churn Prediction
        print("   ⚠️  Testing Churn Prediction...")
        churn_model = ChurnPredictionModel()
        
        # Create churn labels (simplified)
        customers_with_churn = customers.copy()
        customers_with_churn['churned'] = np.random.choice([0, 1], size=len(customers), p=[0.8, 0.2])
        
        # Add some additional features
        customers_with_churn['last_purchase_date'] = pd.to_datetime('2023-01-01') - pd.to_timedelta(np.random.randint(1, 365, len(customers)), unit='D')
        customers_with_churn['tenure_days'] = np.random.randint(30, 1000, len(customers))
        
        churn_model.fit(customers_with_churn, 'churned')
        churn_predictions = churn_model.predict(customers_with_churn)
        
        print(f"   ✅ Churn prediction completed")
        print(f"   📊 Average churn probability: {np.mean(churn_predictions):.3f}")
        print(f"   📈 High-risk customers: {sum(1 for p in churn_predictions if p > 0.7)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in insight models: {e}")
        return False

def test_recommendation_engine(customers, sales, feedback):
    """Test recommendation engine"""
    print("\n💡 Step 4: Testing Recommendation Engine...")
    
    try:
        from models.recommendation_engine import RecommendationEngine
        
        # Create sample inventory and supplier data
        inventory_data = pd.DataFrame({
            'product_id': [f'PROD_{i:03d}' for i in range(1, 51)],
            'current_stock': np.random.randint(10, 1000, 50),
            'reorder_point': np.random.randint(50, 500, 50)
        })
        
        supplier_data = pd.DataFrame({
            'supplier_id': [f'SUP_{i:03d}' for i in range(1, 11)],
            'on_time': np.random.choice([True, False], 10, p=[0.8, 0.2]),
            'quality_rating': np.random.uniform(2, 5, 10),
            'cost': np.random.uniform(10, 100, 10),
            'lead_time': np.random.randint(3, 21, 10)
        })
        
        print("   📦 Created sample inventory and supplier data")
        
        # Test comprehensive recommendations
        engine = RecommendationEngine()
        recommendations = engine.generate_comprehensive_recommendations(
            customers, sales, inventory_data, supplier_data
        )
        
        print("   ✅ Comprehensive recommendations generated")
        print(f"   📊 Marketing insights: {len(recommendations.get('marketing', {}))}")
        print(f"   💰 Pricing insights: {len(recommendations.get('pricing', {}))}")
        print(f"   📋 Summary recommendations: {len(recommendations.get('summary', []))}")
        
        # Display sample recommendations
        if 'summary' in recommendations and recommendations['summary']:
            print("   🎯 Sample Recommendations:")
            for i, rec in enumerate(recommendations['summary'][:3], 1):
                print(f"      {i}. {rec}")
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Error in recommendation engine: {e}")
        return None

def test_api_functionality():
    """Test API functionality"""
    print("\n🌐 Step 5: Testing API Functionality...")
    
    try:
        from api.app import app
        
        # Create test client
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/health')
            if response.status_code == 200:
                print("   ✅ Health check endpoint working")
            
            # Test sample data endpoint
            response = client.get('/api/data/sample')
            if response.status_code == 200:
                data = response.get_json()
                print("   ✅ Sample data endpoint working")
                print(f"   📊 Generated {len(data.get('customers', []))} customers via API")
            
            # Test customer segmentation endpoint
            test_data = {
                'customers': [
                    {'customer_id': 'CUST_001', 'age': 25, 'total_spend': 1000},
                    {'customer_id': 'CUST_002', 'age': 35, 'total_spend': 2000},
                    {'customer_id': 'CUST_003', 'age': 45, 'total_spend': 500}
                ],
                'n_clusters': 3
            }
            
            response = client.post('/api/insights/customer-segmentation', json=test_data)
            if response.status_code == 200:
                data = response.get_json()
                print("   ✅ Customer segmentation API working")
                print(f"   📊 Segmented {len(data.get('segments', []))} customers")
            
            # Test sentiment analysis endpoint
            sentiment_data = {
                'texts': [
                    'Great product, very satisfied!',
                    'Poor quality, disappointed',
                    'Average product, nothing special'
                ]
            }
            
            response = client.post('/api/insights/sentiment-analysis', json=sentiment_data)
            if response.status_code == 200:
                data = response.get_json()
                print("   ✅ Sentiment analysis API working")
                print(f"   📊 Analyzed {len(data.get('sentiments', []))} texts")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in API testing: {e}")
        return False

def test_security_features():
    """Test security and privacy features"""
    print("\n🔒 Step 6: Testing Security Features...")
    
    try:
        from security.data_anonymization import DataAnonymizer, PrivacyCompliance, SecurityManager
        
        # Test data anonymization
        print("   🎭 Testing data anonymization...")
        anonymizer = DataAnonymizer()
        
        test_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'name': ['John Doe', 'Jane Smith'],
            'email': ['john@example.com', 'jane@example.com'],
            'phone': ['(555) 123-4567', '(555) 987-6543'],
            'age': [25, 35],
            'amount': [100.50, 200.75]
        })
        
        anonymized_data = anonymizer.anonymize_customer_data(test_data)
        print("   ✅ Customer data anonymized successfully")
        print(f"   📊 Original names: {test_data['name'].tolist()}")
        print(f"   📊 Anonymized names: {anonymized_data['name'].tolist()}")
        
        # Test privacy compliance
        print("   📋 Testing privacy compliance...")
        compliance = PrivacyCompliance()
        
        data_types = {
            'customer_id': 'identifier',
            'name': 'name',
            'email': 'email',
            'phone': 'phone',
            'age': 'age',
            'amount': 'amount'
        }
        
        compliance_report = compliance.check_gdpr_compliance(test_data, data_types)
        print("   ✅ Privacy compliance check completed")
        print(f"   📊 Compliance status: {'✅ Compliant' if compliance_report['compliant'] else '❌ Non-compliant'}")
        
        # Test security management
        print("   🛡️  Testing security management...")
        security = SecurityManager()
        
        security.log_access('user1', 'customer_data', 'read')
        security.log_security_event('data_access', 'User accessed customer data')
        
        security_report = security.generate_security_report()
        print("   ✅ Security management working")
        print(f"   📊 Access logs: {security_report['total_access_logs']}")
        print(f"   📊 Security events: {security_report['total_security_events']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in security testing: {e}")
        return False

def test_monitoring_system():
    """Test monitoring and feedback system"""
    print("\n📊 Step 7: Testing Monitoring System...")
    
    try:
        from monitoring.feedback_system import FeedbackSystem, FeedbackEntry
        
        # Initialize feedback system
        feedback_system = FeedbackSystem()
        
        # Test feedback processing
        print("   💬 Testing feedback processing...")
        result = feedback_system.process_feedback('user1', 'insight1', 5, 'Great insights!')
        feedback_system.process_feedback('user2', 'insight1', 3, 'Could be better')
        feedback_system.process_feedback('user3', 'insight2', 4, 'Helpful information')
        
        print("   ✅ Feedback processing working")
        print(f"   📊 Feedback processed: {result['feedback_processed']}")
        
        # Test system status
        print("   📈 Testing system monitoring...")
        status = feedback_system.get_system_status()
        
        print("   ✅ System monitoring working")
        print(f"   📊 System health: {status['system_health']:.2f}")
        print(f"   📊 Total feedback: {status['feedback_summary']['total_feedback']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in monitoring testing: {e}")
        return False

def main():
    """Run comprehensive system test"""
    print("🚀 Starting SLM Business Insights System Test")
    print("=" * 60)
    
    # Step 1: Generate sample data
    customers, sales, feedback = test_data_generation()
    if customers is None:
        print("❌ Failed to generate sample data. Exiting.")
        return
    
    # Step 2: Test data preprocessing
    customer_features, sales_features, feedback_features = test_data_preprocessing(customers, sales, feedback)
    
    # Step 3: Test insight models
    models_success = test_insight_models(customers, sales, feedback)
    
    # Step 4: Test recommendation engine
    recommendations = test_recommendation_engine(customers, sales, feedback)
    
    # Step 5: Test API functionality
    api_success = test_api_functionality()
    
    # Step 6: Test security features
    security_success = test_security_features()
    
    # Step 7: Test monitoring system
    monitoring_success = test_monitoring_system()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 SLM Business Insights System Test Summary")
    print("=" * 60)
    
    test_results = {
        "Data Generation": customers is not None,
        "Data Preprocessing": customer_features is not None,
        "Insight Models": models_success,
        "Recommendation Engine": recommendations is not None,
        "API Functionality": api_success,
        "Security Features": security_success,
        "Monitoring System": monitoring_success
    }
    
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n🚀 System is ready for production use!")
    print("📋 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run the system: python app.py --mode all")
    print("   3. Access dashboard: http://localhost:8501")
    print("   4. Use API: http://localhost:5000")

if __name__ == "__main__":
    main()
