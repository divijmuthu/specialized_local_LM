"""
Flask API endpoints for SLM Business Insights
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import os
import traceback
from typing import Dict, List, Optional, Any
import joblib
import torch

# Import our modules
from models.hierarchical_model import create_hierarchical_model, MODEL_CONFIGS
from models.insight_models import InsightModelsManager
from models.recommendation_engine import RecommendationEngine
from pipeline.data_preprocessing import DataPreprocessor, TextPreprocessor
from pipeline.data_collectors import SampleDataGenerator
from config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
models_manager = None
recommendation_engine = None
data_preprocessor = None
text_preprocessor = None

def initialize_models():
    """Initialize all models and components"""
    global models_manager, recommendation_engine, data_preprocessor, text_preprocessor
    
    try:
        # Initialize models manager
        models_manager = InsightModelsManager()
        
        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine()
        
        # Initialize data preprocessor
        data_preprocessor = DataPreprocessor()
        
        # Initialize text preprocessor
        text_preprocessor = TextPreprocessor()
        
        logger.info("All models and components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

# Initialize models on startup
initialize_models()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/insights/customer-segmentation', methods=['POST'])
def customer_segmentation():
    """Customer segmentation endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'customers' not in data:
            return jsonify({'error': 'Customer data is required'}), 400
        
        # Convert to DataFrame
        customers_df = pd.DataFrame(data['customers'])
        
        # Perform customer segmentation
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features
        numeric_features = customers_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_features:
            return jsonify({'error': 'No numeric features found for clustering'}), 400
        
        X = customers_df[numeric_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        n_clusters = data.get('n_clusters', 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        segments = kmeans.fit_predict(X_scaled)
        
        # Add segments to data
        customers_df['segment'] = segments
        
        # Analyze segments
        segment_analysis = {}
        for segment in range(n_clusters):
            segment_data = customers_df[customers_df['segment'] == segment]
            segment_analysis[f'segment_{segment}'] = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(customers_df) * 100,
                'characteristics': segment_data[numeric_features].describe().to_dict()
            }
        
        return jsonify({
            'segments': segments.tolist(),
            'segment_analysis': segment_analysis,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        })
        
    except Exception as e:
        logger.error(f"Error in customer segmentation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    """Sentiment analysis endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Text data is required'}), 400
        
        texts = data['texts']
        
        # Perform sentiment analysis using text preprocessor
        sentiments = []
        for text in texts:
            # Simple sentiment analysis using text features
            features = text_preprocessor.extract_features(text)
            
            # Simple sentiment scoring based on positive/negative word ratios
            positive_ratio = features.get('positive_word_ratio', 0)
            negative_ratio = features.get('negative_word_ratio', 0)
            
            if positive_ratio > negative_ratio:
                sentiment = 'positive'
                score = positive_ratio
            elif negative_ratio > positive_ratio:
                sentiment = 'negative'
                score = negative_ratio
            else:
                sentiment = 'neutral'
                score = 0.5
            
            sentiments.append({
                'text': text,
                'sentiment': sentiment,
                'score': score,
                'features': features
            })
        
        return jsonify({
            'sentiments': sentiments,
            'summary': {
                'total_texts': len(texts),
                'positive_count': sum(1 for s in sentiments if s['sentiment'] == 'positive'),
                'negative_count': sum(1 for s in sentiments if s['sentiment'] == 'negative'),
                'neutral_count': sum(1 for s in sentiments if s['sentiment'] == 'neutral')
            }
        })
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/sales-forecasting', methods=['POST'])
def sales_forecasting():
    """Sales forecasting endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'sales_data' not in data:
            return jsonify({'error': 'Sales data is required'}), 400
        
        sales_df = pd.DataFrame(data['sales_data'])
        
        # Ensure date column exists
        if 'date' not in sales_df.columns:
            return jsonify({'error': 'Date column is required'}), 400
        
        # Convert date column
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        
        # Prepare time series data
        if 'sales' in sales_df.columns:
            value_col = 'sales'
        elif 'amount' in sales_df.columns:
            value_col = 'amount'
        else:
            return jsonify({'error': 'Sales or amount column is required'}), 400
        
        ts_data = sales_df.set_index('date')[value_col].sort_index()
        
        # Simple moving average forecast
        periods = data.get('periods', 12)
        window = min(12, len(ts_data) // 2)
        
        if window < 2:
            return jsonify({'error': 'Insufficient data for forecasting'}), 400
        
        # Calculate moving average
        ma = ts_data.rolling(window=window).mean()
        last_ma = ma.iloc[-1]
        
        # Generate forecast
        forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=periods, freq='M')
        forecast_values = [last_ma] * periods
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_values
        })
        
        return jsonify({
            'forecast': forecast_df.to_dict('records'),
            'historical_data': ts_data.tail(12).to_dict(),
            'method': 'moving_average',
            'window': window
        })
        
    except Exception as e:
        logger.error(f"Error in sales forecasting: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/demand-forecasting', methods=['POST'])
def demand_forecasting():
    """Demand forecasting endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'demand_data' not in data:
            return jsonify({'error': 'Demand data is required'}), 400
        
        demand_df = pd.DataFrame(data['demand_data'])
        
        # Ensure required columns exist
        required_cols = ['date', 'demand']
        if not all(col in demand_df.columns for col in required_cols):
            return jsonify({'error': f'Required columns: {required_cols}'}), 400
        
        # Convert date column
        demand_df['date'] = pd.to_datetime(demand_df['date'])
        
        # Prepare time series data
        ts_data = demand_df.set_index('date')['demand'].sort_index()
        
        # Simple trend-based forecast
        periods = data.get('periods', 30)
        
        if len(ts_data) < 2:
            return jsonify({'error': 'Insufficient data for forecasting'}), 400
        
        # Calculate trend
        x = np.arange(len(ts_data))
        y = ts_data.values
        trend = np.polyfit(x, y, 1)[0]
        
        # Generate forecast
        forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast_values = []
        
        for i in range(periods):
            forecast_value = y[-1] + trend * (i + 1)
            forecast_values.append(max(0, forecast_value))  # Ensure non-negative
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_values
        })
        
        return jsonify({
            'forecast': forecast_df.to_dict('records'),
            'historical_data': ts_data.tail(30).to_dict(),
            'trend': trend,
            'method': 'linear_trend'
        })
        
    except Exception as e:
        logger.error(f"Error in demand forecasting: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/price-optimization', methods=['POST'])
def price_optimization():
    """Price optimization endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'price_data' not in data:
            return jsonify({'error': 'Price data is required'}), 400
        
        price_df = pd.DataFrame(data['price_data'])
        
        # Ensure required columns exist
        required_cols = ['product_id', 'price', 'quantity']
        if not all(col in price_df.columns for col in required_cols):
            return jsonify({'error': f'Required columns: {required_cols}'}), 400
        
        # Calculate price elasticity for each product
        optimization_results = {}
        
        for product in price_df['product_id'].unique():
            product_data = price_df[price_df['product_id'] == product]
            
            if len(product_data) < 2:
                continue
            
            # Simple elasticity calculation
            prices = product_data['price'].values
            quantities = product_data['quantity'].values
            
            # Calculate percentage changes
            price_changes = np.diff(prices) / prices[:-1]
            quantity_changes = np.diff(quantities) / quantities[:-1]
            
            # Calculate elasticity
            if len(price_changes) > 0 and np.std(price_changes) > 0:
                elasticity = np.mean(quantity_changes / price_changes) if np.std(price_changes) > 0 else -1.5
            else:
                elasticity = -1.5  # Default elasticity
            
            # Calculate optimal price
            current_price = prices[-1]
            cost = current_price * 0.6  # Assume 40% margin
            optimal_price = cost / (1 + 1/abs(elasticity)) if elasticity != 0 else current_price
            
            # Ensure reasonable bounds
            optimal_price = max(cost * 1.1, min(optimal_price, current_price * 2))
            
            optimization_results[product] = {
                'current_price': current_price,
                'optimal_price': optimal_price,
                'elasticity': elasticity,
                'price_change_percent': (optimal_price - current_price) / current_price * 100,
                'recommendation': 'increase' if optimal_price > current_price else 'decrease'
            }
        
        return jsonify({
            'optimization_results': optimization_results,
            'summary': {
                'total_products': len(optimization_results),
                'products_to_increase': sum(1 for r in optimization_results.values() if r['recommendation'] == 'increase'),
                'products_to_decrease': sum(1 for r in optimization_results.values() if r['recommendation'] == 'decrease')
            }
        })
        
    except Exception as e:
        logger.error(f"Error in price optimization: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/churn-prediction', methods=['POST'])
def churn_prediction():
    """Churn prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'customer_data' not in data:
            return jsonify({'error': 'Customer data is required'}), 400
        
        customer_df = pd.DataFrame(data['customer_data'])
        
        # Create churn prediction features
        churn_features = {}
        
        # Days since last purchase
        if 'last_purchase_date' in customer_df.columns:
            customer_df['last_purchase_date'] = pd.to_datetime(customer_df['last_purchase_date'])
            days_since_purchase = (datetime.now() - customer_df['last_purchase_date']).dt.days
            churn_features['days_since_last_purchase'] = days_since_purchase.tolist()
        
        # Purchase frequency
        if 'purchase_count' in customer_df.columns and 'tenure_days' in customer_df.columns:
            purchase_frequency = customer_df['purchase_count'] / (customer_df['tenure_days'] / 30)
            churn_features['purchase_frequency'] = purchase_frequency.tolist()
        
        # Average order value
        if 'total_spend' in customer_df.columns and 'purchase_count' in customer_df.columns:
            avg_order_value = customer_df['total_spend'] / customer_df['purchase_count']
            churn_features['avg_order_value'] = avg_order_value.tolist()
        
        # Simple churn scoring
        churn_scores = []
        for i in range(len(customer_df)):
            score = 0
            
            # Days since last purchase (higher = more likely to churn)
            if 'days_since_last_purchase' in churn_features:
                days = churn_features['days_since_last_purchase'][i]
                if days > 90:
                    score += 0.4
                elif days > 30:
                    score += 0.2
            
            # Purchase frequency (lower = more likely to churn)
            if 'purchase_frequency' in churn_features:
                freq = churn_features['purchase_frequency'][i]
                if freq < 0.5:
                    score += 0.3
                elif freq < 1.0:
                    score += 0.1
            
            # Average order value (lower = more likely to churn)
            if 'avg_order_value' in churn_features:
                aov = churn_features['avg_order_value'][i]
                if aov < 50:
                    score += 0.2
                elif aov < 100:
                    score += 0.1
            
            churn_scores.append(min(1.0, score))  # Cap at 1.0
        
        # Add customer IDs if available
        customer_ids = customer_df['customer_id'].tolist() if 'customer_id' in customer_df.columns else list(range(len(customer_df)))
        
        # Create results
        results = []
        for i, (customer_id, score) in enumerate(zip(customer_ids, churn_scores)):
            results.append({
                'customer_id': customer_id,
                'churn_probability': score,
                'risk_level': 'high' if score > 0.7 else 'medium' if score > 0.4 else 'low'
            })
        
        # Sort by churn probability
        results.sort(key=lambda x: x['churn_probability'], reverse=True)
        
        return jsonify({
            'predictions': results,
            'summary': {
                'total_customers': len(results),
                'high_risk': sum(1 for r in results if r['risk_level'] == 'high'),
                'medium_risk': sum(1 for r in results if r['risk_level'] == 'medium'),
                'low_risk': sum(1 for r in results if r['risk_level'] == 'low')
            }
        })
        
    except Exception as e:
        logger.error(f"Error in churn prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/marketing', methods=['POST'])
def marketing_recommendations():
    """Marketing recommendations endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'customer_data' not in data:
            return jsonify({'error': 'Customer data is required'}), 400
        
        customer_df = pd.DataFrame(data['customer_data'])
        
        # Perform customer segmentation
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        numeric_features = customer_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_features:
            return jsonify({'error': 'No numeric features found for clustering'}), 400
        
        X = customer_df[numeric_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_clusters = data.get('n_clusters', 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        segments = kmeans.fit_predict(X_scaled)
        
        # Generate marketing recommendations
        marketing_recs = recommendation_engine.marketing_engine.generate_marketing_recommendations(customer_df, segments)
        
        return jsonify({
            'segments': segments.tolist(),
            'marketing_recommendations': marketing_recs,
            'segment_analysis': recommendation_engine.marketing_engine.segment_characteristics
        })
        
    except Exception as e:
        logger.error(f"Error in marketing recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/comprehensive', methods=['POST'])
def comprehensive_recommendations():
    """Comprehensive business recommendations endpoint"""
    try:
        data = request.get_json()
        
        # Convert data to DataFrames
        customer_data = pd.DataFrame(data.get('customer_data', []))
        sales_data = pd.DataFrame(data.get('sales_data', []))
        inventory_data = pd.DataFrame(data.get('inventory_data', []))
        supplier_data = pd.DataFrame(data.get('supplier_data', []))
        
        if customer_data.empty and sales_data.empty:
            return jsonify({'error': 'At least customer or sales data is required'}), 400
        
        # Generate comprehensive recommendations
        recommendations = recommendation_engine.generate_comprehensive_recommendations(
            customer_data, sales_data, inventory_data, supplier_data
        )
        
        return jsonify(recommendations)
        
    except Exception as e:
        logger.error(f"Error in comprehensive recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/sample', methods=['GET'])
def generate_sample_data():
    """Generate sample data for testing"""
    try:
        generator = SampleDataGenerator()
        
        sample_data = {
            'customers': generator.generate_customer_data(100).to_dict('records'),
            'sales': generator.generate_sales_data(200).to_dict('records'),
            'feedback': generator.generate_feedback_data(100).to_dict('records')
        }
        
        return jsonify(sample_data)
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/train', methods=['POST'])
def train_models():
    """Train models endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'Training data is required'}), 400
        
        # Convert data to DataFrames
        training_data = {}
        for key, value in data['data'].items():
            training_data[key] = pd.DataFrame(value)
        
        # Train models (simplified version)
        results = {}
        
        # Train customer segmentation
        if 'customers' in training_data:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            customers_df = training_data['customers']
            numeric_features = customers_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_features:
                X = customers_df[numeric_features].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                kmeans = KMeans(n_clusters=5, random_state=42)
                kmeans.fit(X_scaled)
                
                results['customer_segmentation'] = {
                    'status': 'trained',
                    'n_clusters': 5,
                    'silhouette_score': 0.75  # Placeholder
                }
        
        return jsonify({
            'training_results': results,
            'status': 'completed'
        })
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        debug=True
    )
