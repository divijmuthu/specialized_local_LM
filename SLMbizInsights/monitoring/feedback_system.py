"""
Continuous learning, monitoring, and feedback loops
"""
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os
import pickle
import sqlite3
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import asyncio
import aiohttp
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEntry:
    """Feedback entry data structure"""
    user_id: str
    insight_id: str
    rating: int  # 1-5 scale
    feedback_text: Optional[str] = None
    timestamp: datetime = None
    model_version: str = "1.0"
    prediction_confidence: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: Optional[float] = None
    r2_score: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class FeedbackCollector:
    """Collect and manage user feedback"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self.init_database()
        self.feedback_buffer = []
        self.buffer_size = 100
    
    def init_database(self):
        """Initialize feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                insight_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                feedback_text TEXT,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prediction_confidence REAL
            )
        ''')
        
        # Create model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                mse REAL,
                r2_score REAL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        # Create data drift table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_drift (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                drift_score REAL NOT NULL,
                p_value REAL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, feedback: FeedbackEntry):
        """Add feedback to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (user_id, insight_id, rating, feedback_text, 
                                timestamp, model_version, prediction_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (feedback.user_id, feedback.insight_id, feedback.rating, 
              feedback.feedback_text, feedback.timestamp.isoformat(),
              feedback.model_version, feedback.prediction_confidence))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback added: {feedback.user_id} rated {feedback.insight_id} as {feedback.rating}")
    
    def get_feedback(self, user_id: Optional[str] = None, 
                    insight_id: Optional[str] = None,
                    days: int = 30) -> List[Dict[str, Any]]:
        """Retrieve feedback from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM feedback WHERE timestamp >= ?"
        params = [(datetime.now() - timedelta(days=days)).isoformat()]
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if insight_id:
            query += " AND insight_id = ?"
            params.append(insight_id)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        conn.close()
        
        # Convert to list of dictionaries
        columns = ['id', 'user_id', 'insight_id', 'rating', 'feedback_text', 
                  'timestamp', 'model_version', 'prediction_confidence']
        
        return [dict(zip(columns, row)) for row in results]
    
    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback summary statistics"""
        feedback_data = self.get_feedback(days=days)
        
        if not feedback_data:
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'rating_distribution': {},
                'response_rate': 0
            }
        
        ratings = [f['rating'] for f in feedback_data]
        
        return {
            'total_feedback': len(feedback_data),
            'average_rating': np.mean(ratings),
            'rating_distribution': {str(i): ratings.count(i) for i in range(1, 6)},
            'response_rate': len(feedback_data) / max(1, len(feedback_data)) * 100
        }

class ModelMonitor:
    """Monitor model performance and detect drift"""
    
    def __init__(self):
        self.performance_history = []
        self.drift_threshold = 0.1
        self.performance_threshold = 0.8
    
    def log_model_performance(self, performance: ModelPerformance):
        """Log model performance metrics"""
        self.performance_history.append(performance)
        
        # Store in database
        conn = sqlite3.connect("feedback.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance (model_name, accuracy, precision, recall, 
                                         f1_score, mse, r2_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (performance.model_name, performance.accuracy, performance.precision,
              performance.recall, performance.f1_score, performance.mse,
              performance.r2_score, performance.timestamp.isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Model performance logged: {performance.model_name} - Accuracy: {performance.accuracy:.3f}")
    
    def detect_performance_degradation(self, model_name: str, 
                                     window_size: int = 10) -> Dict[str, Any]:
        """Detect model performance degradation"""
        model_performance = [p for p in self.performance_history if p.model_name == model_name]
        
        if len(model_performance) < window_size:
            return {
                'degradation_detected': False,
                'message': 'Insufficient data for degradation detection'
            }
        
        recent_performance = model_performance[-window_size:]
        historical_performance = model_performance[:-window_size]
        
        if not historical_performance:
            return {
                'degradation_detected': False,
                'message': 'No historical data for comparison'
            }
        
        recent_accuracy = np.mean([p.accuracy for p in recent_performance])
        historical_accuracy = np.mean([p.accuracy for p in historical_performance])
        
        accuracy_drop = historical_accuracy - recent_accuracy
        
        degradation_detected = accuracy_drop > self.performance_threshold
        
        return {
            'degradation_detected': degradation_detected,
            'accuracy_drop': accuracy_drop,
            'recent_accuracy': recent_accuracy,
            'historical_accuracy': historical_accuracy,
            'recommendation': 'Retrain model' if degradation_detected else 'Continue monitoring'
        }
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between current and reference data"""
        drift_results = {}
        
        for column in current_data.select_dtypes(include=[np.number]).columns:
            if column in reference_data.columns:
                # Kolmogorov-Smirnov test for drift detection
                from scipy import stats
                
                current_values = current_data[column].dropna()
                reference_values = reference_data[column].dropna()
                
                if len(current_values) > 0 and len(reference_values) > 0:
                    statistic, p_value = stats.ks_2samp(reference_values, current_values)
                    
                    drift_results[column] = {
                        'drift_score': statistic,
                        'p_value': p_value,
                        'drift_detected': statistic > self.drift_threshold
                    }
                    
                    # Log drift detection
                    if statistic > self.drift_threshold:
                        logger.warning(f"Data drift detected in {column}: score={statistic:.3f}")
        
        return drift_results
    
    def generate_performance_report(self, model_name: str) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        model_performance = [p for p in self.performance_history if p.model_name == model_name]
        
        if not model_performance:
            return {'error': 'No performance data available'}
        
        # Calculate trends
        accuracies = [p.accuracy for p in model_performance]
        timestamps = [p.timestamp for p in model_performance]
        
        # Performance trend
        if len(accuracies) > 1:
            trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
        else:
            trend = 0
        
        # Recent performance
        recent_performance = model_performance[-5:] if len(model_performance) >= 5 else model_performance
        recent_accuracy = np.mean([p.accuracy for p in recent_performance])
        
        return {
            'model_name': model_name,
            'total_evaluations': len(model_performance),
            'current_accuracy': recent_accuracy,
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies),
            'average_accuracy': np.mean(accuracies),
            'accuracy_trend': trend,
            'performance_stability': 1 - np.std(accuracies) if len(accuracies) > 1 else 1,
            'last_evaluation': model_performance[-1].timestamp.isoformat() if model_performance else None
        }

class ContinuousLearning:
    """Continuous learning and model retraining"""
    
    def __init__(self, feedback_collector: FeedbackCollector, model_monitor: ModelMonitor):
        self.feedback_collector = feedback_collector
        self.model_monitor = model_monitor
        self.retraining_threshold = 0.05  # 5% performance drop
        self.min_feedback_samples = 100
    
    def should_retrain(self, model_name: str) -> Dict[str, Any]:
        """Determine if model should be retrained"""
        # Check performance degradation
        performance_check = self.model_monitor.detect_performance_degradation(model_name)
        
        # Check feedback quality
        feedback_summary = self.feedback_collector.get_feedback_summary()
        
        # Check if we have enough new feedback
        has_sufficient_feedback = feedback_summary['total_feedback'] >= self.min_feedback_samples
        
        # Check feedback quality
        average_rating = feedback_summary['average_rating']
        low_quality_feedback = average_rating < 3.0
        
        should_retrain = (
            performance_check['degradation_detected'] or
            (has_sufficient_feedback and low_quality_feedback)
        )
        
        return {
            'should_retrain': should_retrain,
            'reasons': {
                'performance_degradation': performance_check['degradation_detected'],
                'low_feedback_quality': low_quality_feedback,
                'sufficient_feedback': has_sufficient_feedback
            },
            'performance_check': performance_check,
            'feedback_summary': feedback_summary
        }
    
    def prepare_retraining_data(self, model_name: str) -> Dict[str, Any]:
        """Prepare data for model retraining"""
        # Get recent feedback
        recent_feedback = self.feedback_collector.get_feedback(days=30)
        
        # Get performance history
        model_performance = [p for p in self.model_monitor.performance_history 
                           if p.model_name == model_name]
        
        return {
            'feedback_data': recent_feedback,
            'performance_data': model_performance,
            'retraining_trigger': 'performance_degradation' if model_performance else 'feedback_quality',
            'data_quality_score': self._calculate_data_quality_score(recent_feedback)
        }
    
    def _calculate_data_quality_score(self, feedback_data: List[Dict[str, Any]]) -> float:
        """Calculate data quality score based on feedback"""
        if not feedback_data:
            return 0.0
        
        # Factors affecting data quality
        ratings = [f['rating'] for f in feedback_data]
        text_feedback_count = sum(1 for f in feedback_data if f['feedback_text'])
        
        # Quality score based on rating distribution and text feedback
        rating_quality = np.mean(ratings) / 5.0  # Normalize to 0-1
        text_quality = text_feedback_count / len(feedback_data)
        
        return (rating_quality + text_quality) / 2.0
    
    def retrain_model(self, model_name: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Retrain model with new data"""
        try:
            # This is a placeholder for actual model retraining
            # In practice, you would load the existing model and retrain it
            
            logger.info(f"Starting retraining for model: {model_name}")
            
            # Simulate retraining process
            time.sleep(2)  # Simulate training time
            
            # Generate new performance metrics
            new_performance = ModelPerformance(
                model_name=model_name,
                accuracy=0.85 + np.random.normal(0, 0.02),  # Simulate improvement
                precision=0.82 + np.random.normal(0, 0.02),
                recall=0.88 + np.random.normal(0, 0.02),
                f1_score=0.85 + np.random.normal(0, 0.02)
            )
            
            # Log new performance
            self.model_monitor.log_model_performance(new_performance)
            
            return {
                'success': True,
                'new_performance': new_performance,
                'retraining_time': '2 seconds',
                'improvement': 'Model accuracy improved'
            }
            
        except Exception as e:
            logger.error(f"Error retraining model {model_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class MetricsCollector:
    """Collect and expose metrics for monitoring"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.metrics = {
            'predictions_total': Counter('predictions_total', 'Total number of predictions'),
            'prediction_accuracy': Histogram('prediction_accuracy', 'Prediction accuracy'),
            'model_performance': Gauge('model_performance', 'Model performance score'),
            'feedback_count': Counter('feedback_count', 'Total feedback received'),
            'data_drift_score': Gauge('data_drift_score', 'Data drift detection score')
        }
        self.metrics_server_started = False
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        if not self.metrics_server_started:
            start_http_server(self.port)
            self.metrics_server_started = True
            logger.info(f"Metrics server started on port {self.port}")
    
    def record_prediction(self, model_name: str, accuracy: float):
        """Record prediction metrics"""
        self.metrics['predictions_total'].labels(model=model_name).inc()
        self.metrics['prediction_accuracy'].observe(accuracy)
    
    def record_feedback(self, rating: int):
        """Record feedback metrics"""
        self.metrics['feedback_count'].labels(rating=str(rating)).inc()
    
    def update_model_performance(self, model_name: str, performance: float):
        """Update model performance metric"""
        self.metrics['model_performance'].labels(model=model_name).set(performance)
    
    def update_data_drift(self, feature_name: str, drift_score: float):
        """Update data drift metric"""
        self.metrics['data_drift_score'].labels(feature=feature_name).set(drift_score)

class MonitoringDashboard:
    """Monitoring dashboard for system health"""
    
    def __init__(self, feedback_collector: FeedbackCollector, 
                 model_monitor: ModelMonitor, metrics_collector: MetricsCollector):
        self.feedback_collector = feedback_collector
        self.model_monitor = model_monitor
        self.metrics_collector = metrics_collector
    
    def generate_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        # Get feedback summary
        feedback_summary = self.feedback_collector.get_feedback_summary()
        
        # Get model performance
        model_reports = {}
        for model_name in ['customer_segmentation', 'sentiment_analysis', 'churn_prediction']:
            model_reports[model_name] = self.model_monitor.generate_performance_report(model_name)
        
        # Calculate overall system health
        system_health = self._calculate_system_health(feedback_summary, model_reports)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': system_health,
            'feedback_summary': feedback_summary,
            'model_performance': model_reports,
            'recommendations': self._generate_recommendations(feedback_summary, model_reports)
        }
    
    def _calculate_system_health(self, feedback_summary: Dict[str, Any], 
                               model_reports: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        # Feedback health (0-1)
        feedback_health = min(feedback_summary['average_rating'] / 5.0, 1.0)
        
        # Model performance health (0-1)
        model_scores = []
        for model_name, report in model_reports.items():
            if 'current_accuracy' in report:
                model_scores.append(report['current_accuracy'])
        
        model_health = np.mean(model_scores) if model_scores else 0.5
        
        # Overall health (weighted average)
        overall_health = 0.6 * model_health + 0.4 * feedback_health
        
        return overall_health
    
    def _generate_recommendations(self, feedback_summary: Dict[str, Any], 
                                model_reports: Dict[str, Any]) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        # Feedback-based recommendations
        if feedback_summary['average_rating'] < 3.0:
            recommendations.append("Low user satisfaction - review model predictions and user interface")
        
        if feedback_summary['total_feedback'] < 50:
            recommendations.append("Low feedback volume - consider implementing feedback incentives")
        
        # Model performance recommendations
        for model_name, report in model_reports.items():
            if 'current_accuracy' in report and report['current_accuracy'] < 0.8:
                recommendations.append(f"{model_name} performance below threshold - consider retraining")
            
            if 'accuracy_trend' in report and report['accuracy_trend'] < -0.01:
                recommendations.append(f"{model_name} showing declining performance trend")
        
        return recommendations

class FeedbackSystem:
    """Main feedback system coordinator"""
    
    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.model_monitor = ModelMonitor()
        self.continuous_learning = ContinuousLearning(self.feedback_collector, self.model_monitor)
        self.metrics_collector = MetricsCollector()
        self.monitoring_dashboard = MonitoringDashboard(
            self.feedback_collector, self.model_monitor, self.metrics_collector
        )
        
        # Start metrics server
        self.metrics_collector.start_metrics_server()
    
    def process_feedback(self, user_id: str, insight_id: str, rating: int, 
                        feedback_text: Optional[str] = None) -> Dict[str, Any]:
        """Process user feedback"""
        feedback = FeedbackEntry(
            user_id=user_id,
            insight_id=insight_id,
            rating=rating,
            feedback_text=feedback_text
        )
        
        # Add feedback
        self.feedback_collector.add_feedback(feedback)
        
        # Record metrics
        self.metrics_collector.record_feedback(rating)
        
        # Check if retraining is needed
        retraining_check = self.continuous_learning.should_retrain('general')
        
        return {
            'feedback_processed': True,
            'feedback_id': f"{user_id}_{insight_id}_{feedback.timestamp.isoformat()}",
            'retraining_needed': retraining_check['should_retrain'],
            'retraining_reasons': retraining_check['reasons']
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return self.monitoring_dashboard.generate_system_health_report()
    
    def run_continuous_learning_cycle(self):
        """Run continuous learning cycle"""
        logger.info("Starting continuous learning cycle")
        
        # Check all models for retraining
        models_to_check = ['customer_segmentation', 'sentiment_analysis', 'churn_prediction']
        
        for model_name in models_to_check:
            retraining_check = self.continuous_learning.should_retrain(model_name)
            
            if retraining_check['should_retrain']:
                logger.info(f"Retraining needed for {model_name}")
                
                # Prepare retraining data
                retraining_data = self.continuous_learning.prepare_retraining_data(model_name)
                
                # Retrain model (placeholder)
                retraining_result = self.continuous_learning.retrain_model(
                    model_name, pd.DataFrame()  # Placeholder data
                )
                
                if retraining_result['success']:
                    logger.info(f"Successfully retrained {model_name}")
                else:
                    logger.error(f"Failed to retrain {model_name}: {retraining_result['error']}")

if __name__ == "__main__":
    # Test the feedback system
    feedback_system = FeedbackSystem()
    
    # Simulate some feedback
    feedback_system.process_feedback('user1', 'insight1', 5, 'Great insights!')
    feedback_system.process_feedback('user2', 'insight1', 3, 'Could be better')
    feedback_system.process_feedback('user3', 'insight2', 4, 'Helpful information')
    
    # Get system status
    status = feedback_system.get_system_status()
    print("System Status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Run continuous learning cycle
    feedback_system.run_continuous_learning_cycle()
    
    print("Feedback system test completed successfully!")
