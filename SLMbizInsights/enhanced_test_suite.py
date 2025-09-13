"""
Enhanced Test Suite for SLM Business Insights System
Comprehensive testing including performance, business scenarios, and edge cases
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import unittest
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark a function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return result, execution_time
    
    def add_result(self, test_name, execution_time, success=True):
        """Add benchmark result"""
        self.results[test_name] = {
            'execution_time': execution_time,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_summary(self):
        """Get performance summary"""
        if not self.results:
            return {
                'total_tests': 0,
                'successful_tests': 0,
                'success_rate': "0.0%",
                'average_execution_time': "0.000s",
                'max_execution_time': "0.000s"
            }
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r['success'])
        avg_time = np.mean([r['execution_time'] for r in self.results.values()])
        max_time = max([r['execution_time'] for r in self.results.values()])
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': f"{(successful_tests/total_tests)*100:.1f}%",
            'average_execution_time': f"{avg_time:.3f}s",
            'max_execution_time': f"{max_time:.3f}s"
        }

class BusinessScenarioTester:
    """Test business scenarios and use cases"""
    
    def __init__(self):
        self.scenarios = {}
    
    def test_customer_retention_scenario(self):
        """Test customer retention business scenario"""
        print("🧪 Testing Customer Retention Scenario...")
        
        try:
            # Generate customer data with churn indicators
            customers = pd.DataFrame({
                'customer_id': [f'CUST_{i:06d}' for i in range(1000)],
                'age': np.random.randint(18, 80, 1000),
                'total_spend': np.random.uniform(100, 10000, 1000),
                'purchase_count': np.random.randint(1, 50, 1000),
                'days_since_purchase': np.random.randint(1, 365, 1000),
                'support_tickets': np.random.randint(0, 10, 1000),
                'satisfaction_score': np.random.uniform(1, 5, 1000)
            })
            
            # Create churn labels based on business rules
            customers['churn_risk'] = np.where(
                (customers['days_since_purchase'] > 90) & 
                (customers['satisfaction_score'] < 3) & 
                (customers['support_tickets'] > 5), 
                'High', 
                np.where(
                    (customers['days_since_purchase'] > 60) | 
                    (customers['satisfaction_score'] < 3.5), 
                    'Medium', 
                    'Low'
                )
            )
            
            # Test churn prediction
            from models.insight_models import ChurnPredictionModel
            
            churn_model = ChurnPredictionModel()
            churn_model.fit(customers, 'churn_risk')
            predictions = churn_model.predict(customers)
            
            # Validate business logic
            high_risk_customers = customers[customers['churn_risk'] == 'High']
            high_risk_predictions = predictions[customers['churn_risk'] == 'High']
            
            # Business validation: High-risk customers should have higher churn probability
            avg_high_risk_prob = np.mean(high_risk_predictions)
            avg_low_risk_prob = np.mean(predictions[customers['churn_risk'] == 'Low'])
            
            business_logic_valid = avg_high_risk_prob > avg_low_risk_prob
            
            self.scenarios['customer_retention'] = {
                'success': business_logic_valid,
                'high_risk_customers': len(high_risk_customers),
                'avg_high_risk_prob': avg_high_risk_prob,
                'avg_low_risk_prob': avg_low_risk_prob,
                'business_logic_valid': business_logic_valid
            }
            
            print(f"   ✅ Customer retention scenario: {len(high_risk_customers)} high-risk customers identified")
            print(f"   📊 High-risk probability: {avg_high_risk_prob:.3f} vs Low-risk: {avg_low_risk_prob:.3f}")
            
            return business_logic_valid
            
        except Exception as e:
            print(f"   ❌ Customer retention scenario failed: {e}")
            self.scenarios['customer_retention'] = {'success': False, 'error': str(e)}
            return False
    
    def test_pricing_optimization_scenario(self):
        """Test pricing optimization business scenario"""
        print("🧪 Testing Pricing Optimization Scenario...")
        
        try:
            # Generate product and sales data
            products = pd.DataFrame({
                'product_id': [f'PROD_{i:03d}' for i in range(50)],
                'current_price': np.random.uniform(10, 500, 50),
                'cost': np.random.uniform(5, 250, 50),
                'demand_elasticity': np.random.uniform(-2.5, -0.5, 50),
                'competitor_price': np.random.uniform(8, 450, 50)
            })
            
            # Test price optimization
            from models.insight_models import PriceOptimizationModel
            
            price_model = PriceOptimizationModel()
            price_model.fit(products)
            
            # Test optimization for sample products
            optimization_results = []
            for _, product in products.head(5).iterrows():
                optimal_price = price_model.optimize_price(
                    product['product_id'], 
                    product['cost']
                )
                
                # Business validation: Optimal price should be higher than cost
                margin_valid = optimal_price > product['cost']
                competitive_valid = abs(optimal_price - product['competitor_price']) / product['competitor_price'] < 0.5
                
                optimization_results.append({
                    'product_id': product['product_id'],
                    'current_price': product['current_price'],
                    'optimal_price': optimal_price,
                    'margin_valid': margin_valid,
                    'competitive_valid': competitive_valid
                })
            
            # Calculate business metrics
            avg_margin_improvement = np.mean([
                (r['optimal_price'] - r['current_price']) / r['current_price'] 
                for r in optimization_results
            ])
            
            valid_optimizations = sum(1 for r in optimization_results if r['margin_valid'] and r['competitive_valid'])
            
            self.scenarios['pricing_optimization'] = {
                'success': valid_optimizations >= 3,  # At least 3 valid optimizations
                'total_products_tested': len(optimization_results),
                'valid_optimizations': valid_optimizations,
                'avg_margin_improvement': avg_margin_improvement
            }
            
            print(f"   ✅ Pricing optimization scenario: {valid_optimizations}/{len(optimization_results)} valid optimizations")
            print(f"   📊 Average margin improvement: {avg_margin_improvement*100:.1f}%")
            
            return valid_optimizations >= 3
            
        except Exception as e:
            print(f"   ❌ Pricing optimization scenario failed: {e}")
            self.scenarios['pricing_optimization'] = {'success': False, 'error': str(e)}
            return False
    
    def test_sales_forecasting_scenario(self):
        """Test sales forecasting business scenario"""
        print("🧪 Testing Sales Forecasting Scenario...")
        
        try:
            # Generate time series sales data
            dates = pd.date_range('2023-01-01', periods=365, freq='D')
            sales_data = pd.DataFrame({
                'date': dates,
                'sales': 1000 + 500 * np.sin(np.arange(365) * 2 * np.pi / 365) + 
                         np.random.normal(0, 100, 365) + 
                         np.arange(365) * 2  # Trend
            })
            
            # Test sales forecasting
            from models.insight_models import SalesForecastingModel
            
            forecast_model = SalesForecastingModel(method='arima')
            forecast_model.fit(sales_data, 'date', 'sales')
            
            # Generate forecast
            forecast = forecast_model.predict(periods=30)
            
            # Business validation: Forecast should be reasonable
            recent_sales = sales_data['sales'].tail(30).mean()
            forecast_mean = np.mean(forecast)
            
            # Forecast should be within reasonable range of recent sales
            forecast_reasonable = 0.5 * recent_sales < forecast_mean < 2.0 * recent_sales
            
            # Check for trend consistency
            recent_trend = sales_data['sales'].tail(30).diff().mean()
            forecast_trend = np.diff(forecast).mean()
            trend_consistent = abs(recent_trend - forecast_trend) < abs(recent_trend) * 0.5
            
            self.scenarios['sales_forecasting'] = {
                'success': forecast_reasonable and trend_consistent,
                'recent_sales_avg': recent_sales,
                'forecast_avg': forecast_mean,
                'forecast_reasonable': forecast_reasonable,
                'trend_consistent': trend_consistent,
                'forecast_periods': len(forecast)
            }
            
            print(f"   ✅ Sales forecasting scenario: {len(forecast)} periods forecasted")
            print(f"   📊 Recent sales: ${recent_sales:.2f}, Forecast: ${forecast_mean:.2f}")
            
            return forecast_reasonable and trend_consistent
            
        except Exception as e:
            print(f"   ❌ Sales forecasting scenario failed: {e}")
            self.scenarios['sales_forecasting'] = {'success': False, 'error': str(e)}
            return False
    
    def test_customer_segmentation_scenario(self):
        """Test customer segmentation business scenario"""
        print("🧪 Testing Customer Segmentation Scenario...")
        
        try:
            # Generate diverse customer data
            customers = pd.DataFrame({
                'customer_id': [f'CUST_{i:06d}' for i in range(500)],
                'age': np.random.randint(18, 80, 500),
                'total_spend': np.random.uniform(100, 10000, 500),
                'purchase_count': np.random.randint(1, 50, 500),
                'retention_rate': np.random.uniform(0.1, 0.9, 500),
                'location': np.random.choice(['North', 'South', 'East', 'West'], 500),
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 500)
            })
            
            # Test customer segmentation
            from models.insight_models import CustomerSegmentationModel
            
            segmentation_model = CustomerSegmentationModel(n_clusters=5)
            features = ['age', 'total_spend', 'purchase_count', 'retention_rate']
            segmentation_model.fit(customers, features)
            clusters = segmentation_model.predict(customers)
            
            # Analyze clusters
            analysis = segmentation_model.analyze_clusters(customers)
            
            # Business validation: Clusters should be distinct and meaningful
            cluster_sizes = [len(customers[clusters == i]) for i in range(5)]
            min_cluster_size = min(cluster_sizes)
            max_cluster_size = max(cluster_sizes)
            
            # Clusters should be reasonably balanced
            cluster_balance = max_cluster_size / min_cluster_size < 3
            
            # Check for meaningful differences between clusters
            cluster_means = []
            for i in range(5):
                cluster_data = customers[clusters == i]
                cluster_means.append({
                    'avg_spend': cluster_data['total_spend'].mean(),
                    'avg_age': cluster_data['age'].mean(),
                    'avg_purchases': cluster_data['purchase_count'].mean()
                })
            
            # Calculate variance between clusters
            spend_variance = np.var([cm['avg_spend'] for cm in cluster_means])
            age_variance = np.var([cm['avg_age'] for cm in cluster_means])
            
            meaningful_segments = spend_variance > 1000000 and age_variance > 100
            
            self.scenarios['customer_segmentation'] = {
                'success': cluster_balance and meaningful_segments,
                'cluster_sizes': cluster_sizes,
                'cluster_balance': cluster_balance,
                'meaningful_segments': meaningful_segments,
                'spend_variance': spend_variance,
                'age_variance': age_variance
            }
            
            print(f"   ✅ Customer segmentation scenario: 5 clusters created")
            print(f"   📊 Cluster sizes: {cluster_sizes}")
            print(f"   📈 Spend variance: {spend_variance:.0f}, Age variance: {age_variance:.1f}")
            
            return cluster_balance and meaningful_segments
            
        except Exception as e:
            print(f"   ❌ Customer segmentation scenario failed: {e}")
            self.scenarios['customer_segmentation'] = {'success': False, 'error': str(e)}
            return False
    
    def get_scenario_summary(self):
        """Get summary of all business scenarios"""
        if not self.scenarios:
            return "No scenarios tested"
        
        total_scenarios = len(self.scenarios)
        successful_scenarios = sum(1 for s in self.scenarios.values() if s.get('success', False))
        
        return {
            'total_scenarios': total_scenarios,
            'successful_scenarios': successful_scenarios,
            'success_rate': f"{(successful_scenarios/total_scenarios)*100:.1f}%",
            'scenarios': self.scenarios
        }

class EnhancedTestSuite:
    """Enhanced test suite with performance and business scenario testing"""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        self.scenario_tester = BusinessScenarioTester()
        self.test_results = {}
    
    def run_performance_tests(self):
        """Run performance benchmark tests"""
        print("🚀 Running Performance Tests...")
        print("=" * 50)
        
        # Test 1: Data Generation Performance
        print("📊 Testing Data Generation Performance...")
        try:
            from pipeline.data_collectors import SampleDataGenerator
            
            generator = SampleDataGenerator()
            
            # Benchmark customer data generation
            result, exec_time = self.benchmark.benchmark_function(
                generator.generate_customer_data, 1000
            )
            self.benchmark.add_result('customer_data_generation', exec_time, len(result) == 1000)
            print(f"   ✅ Customer data (1000 records): {exec_time:.3f}s")
            
            # Benchmark sales data generation
            result, exec_time = self.benchmark.benchmark_function(
                generator.generate_sales_data, 2000
            )
            self.benchmark.add_result('sales_data_generation', exec_time, len(result) == 2000)
            print(f"   ✅ Sales data (2000 records): {exec_time:.3f}s")
            
            # Benchmark feedback data generation
            result, exec_time = self.benchmark.benchmark_function(
                generator.generate_feedback_data, 1000
            )
            self.benchmark.add_result('feedback_data_generation', exec_time, len(result) == 1000)
            print(f"   ✅ Feedback data (1000 records): {exec_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Data generation performance test failed: {e}")
        
        # Test 2: Model Training Performance
        print("\n🤖 Testing Model Training Performance...")
        try:
            from models.insight_models import CustomerSegmentationModel
            
            # Generate test data
            customers = pd.DataFrame({
                'customer_id': [f'CUST_{i:06d}' for i in range(500)],
                'age': np.random.randint(18, 80, 500),
                'total_spend': np.random.uniform(100, 10000, 500),
                'purchase_count': np.random.randint(1, 50, 500),
                'retention_rate': np.random.uniform(0.1, 0.9, 500)
            })
            
            # Benchmark customer segmentation
            segmentation_model = CustomerSegmentationModel(n_clusters=5)
            features = ['age', 'total_spend', 'purchase_count', 'retention_rate']
            
            result, exec_time = self.benchmark.benchmark_function(
                segmentation_model.fit, customers, features
            )
            self.benchmark.add_result('customer_segmentation_training', exec_time, True)
            print(f"   ✅ Customer segmentation training: {exec_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Model training performance test failed: {e}")
        
        # Test 3: API Response Performance
        print("\n🌐 Testing API Response Performance...")
        try:
            from api.app import app
            
            with app.test_client() as client:
                # Benchmark health check
                result, exec_time = self.benchmark.benchmark_function(
                    client.get, '/health'
                )
                self.benchmark.add_result('api_health_check', exec_time, result.status_code == 200)
                print(f"   ✅ API health check: {exec_time:.3f}s")
                
                # Benchmark sample data generation
                result, exec_time = self.benchmark.benchmark_function(
                    client.get, '/api/data/sample'
                )
                self.benchmark.add_result('api_sample_data', exec_time, result.status_code == 200)
                print(f"   ✅ API sample data: {exec_time:.3f}s")
                
        except Exception as e:
            print(f"   ❌ API performance test failed: {e}")
    
    def run_business_scenario_tests(self):
        """Run business scenario tests"""
        print("\n🎯 Running Business Scenario Tests...")
        print("=" * 50)
        
        # Test customer retention scenario
        self.scenario_tester.test_customer_retention_scenario()
        
        # Test pricing optimization scenario
        self.scenario_tester.test_pricing_optimization_scenario()
        
        # Test sales forecasting scenario
        self.scenario_tester.test_sales_forecasting_scenario()
        
        # Test customer segmentation scenario
        self.scenario_tester.test_customer_segmentation_scenario()
    
    def run_edge_case_tests(self):
        """Run edge case and error handling tests"""
        print("\n⚠️  Running Edge Case Tests...")
        print("=" * 50)
        
        edge_cases = {
            'empty_data': self.test_empty_data_handling,
            'invalid_data': self.test_invalid_data_handling,
            'large_data': self.test_large_data_handling,
            'missing_features': self.test_missing_features_handling
        }
        
        for test_name, test_func in edge_cases.items():
            try:
                print(f"🧪 Testing {test_name.replace('_', ' ').title()}...")
                success = test_func()
                self.test_results[test_name] = {'success': success}
                print(f"   {'✅' if success else '❌'} {test_name}: {'PASSED' if success else 'FAILED'}")
            except Exception as e:
                print(f"   ❌ {test_name}: ERROR - {e}")
                self.test_results[test_name] = {'success': False, 'error': str(e)}
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        try:
            from models.insight_models import CustomerSegmentationModel
            
            # Test with empty DataFrame
            empty_df = pd.DataFrame()
            model = CustomerSegmentationModel(n_clusters=3)
            
            # Should handle empty data gracefully
            try:
                model.fit(empty_df, [])
                return False  # Should raise an exception
            except (ValueError, IndexError):
                return True  # Expected behavior
            
        except Exception:
            return False
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        try:
            from models.insight_models import SentimentAnalysisModel
            
            # Test with invalid text data
            invalid_texts = [None, "", "   ", 123, [], {}]
            model = SentimentAnalysisModel()
            
            for text in invalid_texts:
                try:
                    sentiment, score = model.analyze_sentiment(text)
                    # Should return neutral sentiment for invalid inputs
                    if sentiment != 'neutral':
                        return False
                except Exception:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def test_large_data_handling(self):
        """Test handling of large datasets"""
        try:
            from pipeline.data_collectors import SampleDataGenerator
            
            # Generate large dataset
            generator = SampleDataGenerator()
            large_customers = generator.generate_customer_data(10000)
            
            # Should handle large data without memory issues
            return len(large_customers) == 10000
            
        except Exception:
            return False
    
    def test_missing_features_handling(self):
        """Test handling of missing features"""
        try:
            from models.insight_models import CustomerSegmentationModel
            
            # Create data with missing values
            customers = pd.DataFrame({
                'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
                'age': [25, None, 35],
                'total_spend': [1000, 2000, None],
                'purchase_count': [5, 10, 15]
            })
            
            model = CustomerSegmentationModel(n_clusters=2)
            
            # Should handle missing values gracefully
            try:
                model.fit(customers, ['age', 'total_spend', 'purchase_count'])
                return True
            except Exception:
                return False
            
        except Exception:
            return False
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 ENHANCED TEST SUITE REPORT")
        print("=" * 60)
        
        # Performance Summary
        perf_summary = self.benchmark.get_summary()
        print(f"\n🚀 Performance Summary:")
        print(f"   Total Tests: {perf_summary['total_tests']}")
        print(f"   Success Rate: {perf_summary['success_rate']}")
        print(f"   Average Execution Time: {perf_summary['average_execution_time']}")
        print(f"   Max Execution Time: {perf_summary['max_execution_time']}")
        
        # Business Scenario Summary
        scenario_summary = self.scenario_tester.get_scenario_summary()
        print(f"\n🎯 Business Scenario Summary:")
        print(f"   Total Scenarios: {scenario_summary['total_scenarios']}")
        print(f"   Success Rate: {scenario_summary['success_rate']}")
        
        # Edge Case Summary
        edge_case_success = sum(1 for r in self.test_results.values() if r.get('success', False))
        edge_case_total = len(self.test_results)
        print(f"\n⚠️  Edge Case Summary:")
        print(f"   Total Tests: {edge_case_total}")
        print(f"   Success Rate: {(edge_case_success/edge_case_total)*100:.1f}%" if edge_case_total > 0 else "   No edge case tests run")
        
        # Overall Assessment
        total_tests = perf_summary['total_tests'] + scenario_summary['total_scenarios'] + edge_case_total
        total_success = perf_summary['successful_tests'] + scenario_summary['successful_scenarios'] + edge_case_success
        
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🏆 Overall Assessment:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful Tests: {total_success}")
        print(f"   Overall Success Rate: {overall_success_rate:.1f}%")
        
        if overall_success_rate >= 90:
            print("   🎉 EXCELLENT: System is performing exceptionally well!")
        elif overall_success_rate >= 80:
            print("   ✅ GOOD: System is performing well with minor issues.")
        elif overall_success_rate >= 70:
            print("   ⚠️  FAIR: System needs some improvements.")
        else:
            print("   ❌ POOR: System requires significant improvements.")
        
        return {
            'performance': perf_summary,
            'scenarios': scenario_summary,
            'edge_cases': self.test_results,
            'overall_success_rate': overall_success_rate
        }
    
    def run_all_tests(self):
        """Run all enhanced tests"""
        print("🚀 SLM Business Insights - Enhanced Test Suite")
        print("=" * 60)
        
        # Run performance tests
        self.run_performance_tests()
        
        # Run business scenario tests
        self.run_business_scenario_tests()
        
        # Run edge case tests
        self.run_edge_case_tests()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        return report

def main():
    """Run the enhanced test suite"""
    test_suite = EnhancedTestSuite()
    report = test_suite.run_all_tests()
    
    # Save report to file
    with open('enhanced_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: enhanced_test_report.json")
    
    return report

if __name__ == "__main__":
    main()
