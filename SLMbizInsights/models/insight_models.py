"""
Specialized models for different business insights
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import silhouette_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize, linprog
from mlxtend.frequent_patterns import apriori, association_rules
from surprise import Dataset, KNNBasic, SVD
from stellargraph.layer import GraphSAGE
from stellargraph import StellarGraph
import gym
from stable_baselines3 import PPO
from dowhy import CausalModel
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
import os
from config import Config

logger = logging.getLogger(__name__)

class CustomerSegmentationModel:
    """Model 1: Customer Segmentation using KMeans Clustering"""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, data: pd.DataFrame, features: List[str] = None):
        """Fit the clustering model"""
        if features is None:
            # Use numeric columns
            features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_names = features
        X = data[features].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KMeans
        self.kmeans.fit(X_scaled)
        
        # Calculate silhouette score
        labels = self.kmeans.labels_
        silhouette_avg = silhouette_score(X_scaled, labels)
        logger.info(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict customer segments"""
        X = data[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def get_cluster_centers(self) -> pd.DataFrame:
        """Get cluster centers"""
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        return pd.DataFrame(centers, columns=self.feature_names)
    
    def analyze_clusters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        labels = self.predict(data)
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        analysis = {}
        for cluster in range(self.n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster]
            analysis[f'cluster_{cluster}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
                'characteristics': cluster_data.describe().to_dict()
            }
        
        return analysis

class SentimentAnalysisModel:
    """Model 2: Sentiment Analysis using BERT"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3  # Positive, Negative, Neutral
        )
        self.trainer = None
    
    def prepare_data(self, texts: List[str], labels: List[int]):
        """Prepare data for training"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding=True, 
                max_length=512
            )
        
        # Create dataset
        dataset = pd.DataFrame({'text': texts, 'labels': labels})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def fit(self, texts: List[str], labels: List[int]):
        """Train the sentiment analysis model"""
        # Prepare data
        dataset = self.prepare_data(texts, labels)
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        
        # Train
        self.trainer.train()
        
        return self
    
    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment for texts"""
        if self.trainer is None:
            raise ValueError("Model must be trained first")
        
        predictions = self.trainer.predict(texts)
        return predictions.predictions.argmax(axis=1)

class SalesForecastingModel:
    """Model 3: Sales Forecasting using ARIMA and Prophet"""
    
    def __init__(self, method: str = 'arima'):
        self.method = method
        self.model = None
        self.fitted_model = None
    
    def fit(self, data: pd.DataFrame, date_col: str = 'date', value_col: str = 'sales'):
        """Fit the forecasting model"""
        # Prepare time series data
        ts_data = data.set_index(date_col)[value_col].sort_index()
        
        if self.method == 'arima':
            # ARIMA model
            self.model = ARIMA(ts_data, order=(5, 1, 0))
            self.fitted_model = self.model.fit()
            
        elif self.method == 'prophet':
            # Prophet model
            prophet_data = pd.DataFrame({
                'ds': ts_data.index,
                'y': ts_data.values
            })
            
            self.model = Prophet()
            self.fitted_model = self.model.fit(prophet_data)
        
        return self
    
    def predict(self, periods: int = 12) -> pd.DataFrame:
        """Make predictions"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        if self.method == 'arima':
            forecast = self.fitted_model.forecast(steps=periods)
            return pd.DataFrame({
                'date': pd.date_range(start=pd.Timestamp.now(), periods=periods, freq='M'),
                'forecast': forecast
            })
        
        elif self.method == 'prophet':
            future = self.fitted_model.make_future_dataframe(periods=periods)
            forecast = self.fitted_model.predict(future)
            return forecast[['ds', 'yhat']].tail(periods)

class DemandForecastingModel:
    """Model 4: Demand Forecasting using XGBoost and LightGBM"""
    
    def __init__(self, method: str = 'xgboost'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for demand forecasting"""
        df = data.copy()
        
        # Time features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['date'].dt.dayofweek >= 5
        
        # Lag features
        for lag in [1, 7, 30]:
            df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
        
        # Rolling features
        for window in [7, 30]:
            df[f'demand_rolling_mean_{window}'] = df['demand'].rolling(window=window).mean()
            df[f'demand_rolling_std_{window}'] = df['demand'].rolling(window=window).std()
        
        return df
    
    def fit(self, data: pd.DataFrame, target_col: str = 'demand'):
        """Fit the demand forecasting model"""
        # Create features
        df_features = self.create_features(data)
        
        # Prepare features and target
        feature_cols = [col for col in df_features.columns if col not in [target_col, 'date']]
        X = df_features[feature_cols].fillna(0)
        y = df_features[target_col].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        if self.method == 'xgboost':
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif self.method == 'lightgbm':
            self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model performance - MSE: {mse:.3f}, R²: {r2:.3f}")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        df_features = self.create_features(data)
        feature_cols = [col for col in df_features.columns if col not in ['demand', 'date']]
        X = df_features[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)

class PriceOptimizationModel:
    """Model 5: Price Optimization using Linear Programming"""
    
    def __init__(self):
        self.optimal_prices = {}
        self.demand_elasticity = {}
    
    def fit(self, data: pd.DataFrame):
        """Fit price optimization model"""
        # Estimate demand elasticity for each product
        for product in data['product_id'].unique():
            product_data = data[data['product_id'] == product]
            
            # Simple linear demand model: demand = a - b * price
            X = product_data[['price']].values
            y = product_data['demand'].values
            
            if len(X) > 1:
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate elasticity
                avg_price = product_data['price'].mean()
                avg_demand = product_data['demand'].mean()
                elasticity = -model.coef_[0] * avg_price / avg_demand
                
                self.demand_elasticity[product] = elasticity
        
        return self
    
    def optimize_price(self, product_id: str, cost: float, max_price: float = 1000) -> float:
        """Optimize price for a specific product"""
        if product_id not in self.demand_elasticity:
            return cost * 1.5  # Default markup
        
        elasticity = self.demand_elasticity[product_id]
        
        def profit_function(price):
            # Demand function: Q = Q0 * (P/P0)^(-elasticity)
            # Revenue = P * Q
            # Profit = (P - cost) * Q
            demand = 1000 * (price / 50) ** (-elasticity)  # Simplified demand function
            profit = (price - cost) * demand
            return -profit  # Minimize negative profit
        
        # Optimize
        result = minimize(profit_function, x0=cost * 1.5, bounds=[(cost, max_price)])
        return result.x[0]
    
    def optimize_all_prices(self, product_costs: Dict[str, float]) -> Dict[str, float]:
        """Optimize prices for all products"""
        optimal_prices = {}
        
        for product_id, cost in product_costs.items():
            optimal_prices[product_id] = self.optimize_price(product_id, cost)
        
        return optimal_prices

class ChurnPredictionModel:
    """Model 6: Churn Prediction using Random Forest"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for churn prediction"""
        df = data.copy()
        
        # Customer behavior features
        if 'last_purchase_date' in df.columns:
            df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
            df['days_since_last_purchase'] = (pd.Timestamp.now() - df['last_purchase_date']).dt.days
        
        # Purchase frequency
        if 'purchase_count' in df.columns and 'tenure_days' in df.columns:
            df['purchase_frequency'] = df['purchase_count'] / (df['tenure_days'] / 30)
        
        # Average order value
        if 'total_spend' in df.columns and 'purchase_count' in df.columns:
            df['avg_order_value'] = df['total_spend'] / df['purchase_count']
        
        # Support ticket count (if available)
        if 'support_tickets' in df.columns:
            df['support_ticket_rate'] = df['support_tickets'] / (df['tenure_days'] / 30)
        
        return df
    
    def fit(self, data: pd.DataFrame, target_col: str = 'churned'):
        """Fit the churn prediction model"""
        # Create features
        df_features = self.create_features(data)
        
        # Prepare features and target
        feature_cols = [col for col in df_features.columns if col not in [target_col, 'customer_id', 'last_purchase_date']]
        self.feature_names = feature_cols
        
        X = df_features[feature_cols].fillna(0)
        y = df_features[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features for churn prediction:")
        logger.info(feature_importance.head(10))
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict churn probability"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        df_features = self.create_features(data)
        X = df_features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)[:, 1]  # Probability of churn

class MarketBasketAnalysisModel:
    """Model 7: Market Basket Analysis using Apriori Algorithm"""
    
    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = None
        self.association_rules = None
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for market basket analysis"""
        # Create binary matrix for transactions
        basket_data = data.groupby(['transaction_id', 'product_id']).size().unstack().fillna(0)
        basket_data = (basket_data > 0).astype(int)
        
        return basket_data
    
    def fit(self, data: pd.DataFrame):
        """Fit the market basket analysis model"""
        # Prepare data
        basket_data = self.prepare_data(data)
        
        # Find frequent itemsets
        self.frequent_itemsets = apriori(basket_data, min_support=self.min_support, use_colnames=True)
        
        # Generate association rules
        if len(self.frequent_itemsets) > 0:
            self.association_rules = association_rules(
                self.frequent_itemsets, 
                metric="confidence", 
                min_threshold=self.min_confidence
            )
        
        logger.info(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        if self.association_rules is not None:
            logger.info(f"Generated {len(self.association_rules)} association rules")
        
        return self
    
    def get_top_rules(self, n: int = 10) -> pd.DataFrame:
        """Get top association rules by lift"""
        if self.association_rules is None:
            return pd.DataFrame()
        
        return self.association_rules.nlargest(n, 'lift')
    
    def recommend_products(self, purchased_products: List[str]) -> List[str]:
        """Recommend products based on association rules"""
        if self.association_rules is None:
            return []
        
        recommendations = []
        
        for _, rule in self.association_rules.iterrows():
            antecedents = set(rule['antecedents'])
            consequents = set(rule['consequents'])
            
            if antecedents.issubset(set(purchased_products)):
                recommendations.extend(list(consequents))
        
        # Remove already purchased products
        recommendations = [p for p in recommendations if p not in purchased_products]
        
        return list(set(recommendations))

class SupplyChainOptimizationModel:
    """Model 8: Supply Chain Optimization using Linear Programming"""
    
    def __init__(self):
        self.optimal_solution = None
    
    def optimize_supply_chain(self, 
                            suppliers: List[str],
                            products: List[str],
                            demand: Dict[str, float],
                            capacity: Dict[str, float],
                            costs: Dict[Tuple[str, str], float]) -> Dict[str, Any]:
        """Optimize supply chain using linear programming"""
        
        # Create cost vector (minimize total cost)
        c = []
        variable_names = []
        
        for supplier in suppliers:
            for product in products:
                c.append(costs.get((supplier, product), 1000))  # High cost for unavailable combinations
                variable_names.append(f"{supplier}_{product}")
        
        # Constraint matrix and bounds
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []
        
        # Demand constraints (equality)
        for i, product in enumerate(products):
            constraint = [0] * len(c)
            for j, supplier in enumerate(suppliers):
                constraint[i * len(suppliers) + j] = 1
            A_eq.append(constraint)
            b_eq.append(demand.get(product, 0))
        
        # Capacity constraints (inequality)
        for j, supplier in enumerate(suppliers):
            constraint = [0] * len(c)
            for i, product in enumerate(products):
                constraint[i * len(suppliers) + j] = 1
            A_ub.append(constraint)
            b_ub.append(capacity.get(supplier, 0))
        
        # Solve linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=(0, None), method='highs')
        
        if result.success:
            # Parse solution
            solution = {}
            for i, var_name in enumerate(variable_names):
                supplier, product = var_name.split('_')
                if (supplier, product) not in solution:
                    solution[(supplier, product)] = 0
                solution[(supplier, product)] += result.x[i]
            
            self.optimal_solution = {
                'total_cost': result.fun,
                'allocation': solution,
                'status': 'optimal'
            }
        else:
            self.optimal_solution = {
                'status': 'infeasible',
                'message': result.message
            }
        
        return self.optimal_solution

class NLPInsightsModel:
    """Model 9: NLP Insights using BERT and DistilBERT"""
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def extract_insights(self, texts: List[str]) -> Dict[str, Any]:
        """Extract insights from text data"""
        insights = {}
        
        # Topic modeling using TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top terms
        top_terms = []
        for i in range(min(10, tfidf_matrix.shape[0])):
            doc_scores = tfidf_matrix[i].toarray().flatten()
            top_indices = doc_scores.argsort()[-10:][::-1]
            top_terms.extend([feature_names[idx] for idx in top_indices])
        
        insights['top_terms'] = list(set(top_terms))[:20]
        
        # Sentiment distribution
        from textblob import TextBlob
        sentiments = [TextBlob(text).sentiment.polarity for text in texts]
        insights['sentiment_distribution'] = {
            'positive': sum(1 for s in sentiments if s > 0.1),
            'negative': sum(1 for s in sentiments if s < -0.1),
            'neutral': sum(1 for s in sentiments if -0.1 <= s <= 0.1)
        }
        
        # Text statistics
        insights['text_stats'] = {
            'avg_length': np.mean([len(text.split()) for text in texts]),
            'total_documents': len(texts),
            'unique_words': len(set(' '.join(texts).split()))
        }
        
        return insights

class AnomalyDetectionModel:
    """Model 10: Anomaly Detection using Isolation Forest"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
    
    def fit(self, data: pd.DataFrame, features: List[str] = None):
        """Fit the anomaly detection model"""
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = data[features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled)
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict anomalies"""
        features = [col for col in data.columns if col in self.scaler.feature_names_in_]
        X = data[features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def get_anomaly_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores"""
        features = [col for col in data.columns if col in self.scaler.feature_names_in_]
        X = data[features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.decision_function(X_scaled)

class RecommenderSystemModel:
    """Model 11: Recommender System using Collaborative Filtering"""
    
    def __init__(self, method: str = 'knn'):
        self.method = method
        self.model = None
        self.trainset = None
    
    def prepare_data(self, data: pd.DataFrame) -> Dataset:
        """Prepare data for collaborative filtering"""
        # Create user-item matrix
        user_item_data = data[['user_id', 'item_id', 'rating']].copy()
        
        # Convert to Surprise format
        reader = surprise.Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(user_item_data, reader)
        
        return dataset
    
    def fit(self, data: pd.DataFrame):
        """Fit the recommender system"""
        dataset = self.prepare_data(data)
        self.trainset = dataset.build_full_trainset()
        
        if self.method == 'knn':
            self.model = KNNBasic(sim_options={'user_based': True})
        elif self.method == 'svd':
            self.model = SVD()
        
        self.model.fit(self.trainset)
        
        return self
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Get all items
        all_items = [item for item in self.trainset.all_items()]
        
        # Get user's rated items
        user_items = [item for item in all_items if self.trainset.knows_user(user_id)]
        
        # Predict ratings for unrated items
        recommendations = []
        for item in all_items:
            if item not in user_items:
                pred = self.model.predict(user_id, item)
                recommendations.append((item, pred.est))
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]

class TimeSeriesAnalysisModel:
    """Model 12: Time Series Analysis using LSTM and Prophet"""
    
    def __init__(self, method: str = 'prophet'):
        self.method = method
        self.model = None
    
    def fit(self, data: pd.DataFrame, date_col: str = 'date', value_col: str = 'value'):
        """Fit the time series model"""
        ts_data = data.set_index(date_col)[value_col].sort_index()
        
        if self.method == 'prophet':
            prophet_data = pd.DataFrame({
                'ds': ts_data.index,
                'y': ts_data.values
            })
            
            self.model = Prophet()
            self.model.fit(prophet_data)
        
        return self
    
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

class GraphNeuralNetworkModel:
    """Model 13: Graph Neural Networks using GraphSAGE"""
    
    def __init__(self):
        self.model = None
        self.graph = None
    
    def create_graph(self, nodes: pd.DataFrame, edges: pd.DataFrame) -> StellarGraph:
        """Create graph from nodes and edges"""
        self.graph = StellarGraph(nodes=nodes, edges=edges)
        return self.graph
    
    def fit(self, graph: StellarGraph, node_features: List[str]):
        """Fit the GraphSAGE model"""
        # Create GraphSAGE generator
        generator = GraphSAGELinkGenerator(graph, batch_size=512, num_samples=[10, 5])
        
        # Create GraphSAGE model
        self.model = GraphSAGE(layer_sizes=[32, 32], generator=generator)
        
        # Train model (simplified - in practice, you'd need proper training data)
        # This is a placeholder for the actual training process
        logger.info("GraphSAGE model created (training would require proper setup)")
        
        return self

class ReinforcementLearningModel:
    """Model 14: Reinforcement Learning using PPO"""
    
    def __init__(self, env_name: str = 'CartPole-v1'):
        self.env_name = env_name
        self.model = None
        self.env = None
    
    def create_environment(self):
        """Create RL environment"""
        self.env = gym.make(self.env_name)
        return self.env
    
    def fit(self, total_timesteps: int = 10000):
        """Train the RL model"""
        if self.env is None:
            self.create_environment()
        
        # Create PPO model
        self.model = PPO('MlpPolicy', self.env, verbose=1)
        
        # Train model
        self.model.learn(total_timesteps=total_timesteps)
        
        return self
    
    def predict(self, obs: np.ndarray) -> int:
        """Make prediction using trained model"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        action, _ = self.model.predict(obs)
        return action

class CausalInferenceModel:
    """Model 15: Causal Inference using DoWhy"""
    
    def __init__(self):
        self.model = None
        self.identified_estimand = None
    
    def fit(self, data: pd.DataFrame, 
            treatment: str, 
            outcome: str, 
            common_causes: List[str]):
        """Fit the causal inference model"""
        # Create causal model
        self.model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes
        )
        
        # Identify estimand
        self.identified_estimand = self.model.identify_effect()
        
        return self
    
    def estimate_effect(self, method: str = "backdoor.propensity_score_stratification"):
        """Estimate causal effect"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        estimate = self.model.estimate_effect(
            self.identified_estimand,
            method_name=method
        )
        
        return estimate

class InsightModelsManager:
    """Manager class for all insight models"""
    
    def __init__(self):
        self.models = {
            'customer_segmentation': CustomerSegmentationModel(),
            'sentiment_analysis': SentimentAnalysisModel(),
            'sales_forecasting': SalesForecastingModel(),
            'demand_forecasting': DemandForecastingModel(),
            'price_optimization': PriceOptimizationModel(),
            'churn_prediction': ChurnPredictionModel(),
            'market_basket_analysis': MarketBasketAnalysisModel(),
            'supply_chain_optimization': SupplyChainOptimizationModel(),
            'nlp_insights': NLPInsightsModel(),
            'anomaly_detection': AnomalyDetectionModel(),
            'recommender_system': RecommenderSystemModel(),
            'time_series_analysis': TimeSeriesAnalysisModel(),
            'graph_neural_network': GraphNeuralNetworkModel(),
            'reinforcement_learning': ReinforcementLearningModel(),
            'causal_inference': CausalInferenceModel()
        }
    
    def train_model(self, model_name: str, data: Dict[str, pd.DataFrame], **kwargs):
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        # Model-specific training logic
        if model_name == 'customer_segmentation':
            model.fit(data['customers'])
        elif model_name == 'sentiment_analysis':
            model.fit(data['feedback']['text'].tolist(), data['feedback']['rating'].tolist())
        elif model_name == 'sales_forecasting':
            model.fit(data['sales'])
        # Add more model-specific training logic as needed
        
        return model
    
    def get_insights(self, model_name: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get insights from a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        # Model-specific insight generation
        if model_name == 'customer_segmentation':
            return model.analyze_clusters(data['customers'])
        elif model_name == 'nlp_insights':
            return model.extract_insights(data['feedback']['text'].tolist())
        # Add more model-specific insight logic as needed
        
        return {}
    
    def save_models(self, output_dir: str):
        """Save all trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if hasattr(model, 'model') and model.model is not None:
                filepath = os.path.join(output_dir, f"{name}_model.pkl")
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} model to {filepath}")
    
    def load_models(self, input_dir: str):
        """Load all trained models"""
        for name in self.models.keys():
            filepath = os.path.join(input_dir, f"{name}_model.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.models[name] = pickle.load(f)
                logger.info(f"Loaded {name} model from {filepath}")

if __name__ == "__main__":
    # Test the insight models
    from pipeline.data_collectors import SampleDataGenerator
    
    # Generate sample data
    generator = SampleDataGenerator()
    sample_data = {
        'customers': generator.generate_customer_data(1000),
        'sales': generator.generate_sales_data(2000),
        'feedback': generator.generate_feedback_data(1000)
    }
    
    # Test customer segmentation
    segmentation_model = CustomerSegmentationModel()
    segmentation_model.fit(sample_data['customers'])
    clusters = segmentation_model.predict(sample_data['customers'])
    analysis = segmentation_model.analyze_clusters(sample_data['customers'])
    
    print("Customer Segmentation Results:")
    print(f"Number of clusters: {len(set(clusters))}")
    print("Cluster analysis:", analysis)
    
    # Test sentiment analysis
    sentiment_model = SentimentAnalysisModel()
    # Note: This would require actual training in a real scenario
    
    print("Insight models test completed successfully!")
