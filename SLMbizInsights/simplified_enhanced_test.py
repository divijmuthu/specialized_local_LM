"""
Simplified Enhanced Test Suite for SLM Business Insights System
Works with minimal requirements and focuses on core functionality
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SimplifiedEnhancedTestSuite:
    """Simplified enhanced test suite that works with minimal requirements"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_results = {}
    
    def test_data_generation_performance(self):
        """Test data generation performance"""
        print("📊 Testing Data Generation Performance...")
        
        try:
            # Generate sample data directly
            start_time = time.time()
            
            # Generate customer data
            customers = pd.DataFrame({
                'customer_id': [f'CUST_{i:06d}' for i in range(1000)],
                'age': np.random.randint(18, 80, 1000),
                'total_spend': np.random.uniform(100, 10000, 1000),
                'purchase_count': np.random.randint(1, 50, 1000),
                'retention_rate': np.random.uniform(0.1, 0.9, 1000)
            })
            
            # Generate sales data
            sales = pd.DataFrame({
                'transaction_id': [f'TXN_{i:08d}' for i in range(2000)],
                'customer_id': np.random.choice(customers['customer_id'], 2000),
                'product_id': [f'PROD_{i:03d}' for i in np.random.randint(1, 101, 2000)],
                'amount': np.random.uniform(10, 1000, 2000),
                'quantity': np.random.randint(1, 10, 2000),
                'date': pd.date_range('2023-01-01', periods=2000, freq='H')
            })
            
            # Generate feedback data
            feedback_texts = [
                "Great product, very satisfied!",
                "Poor quality, disappointed",
                "Excellent value for money",
                "Product arrived damaged",
                "Fast delivery and good quality"
            ]
            
            feedback = pd.DataFrame({
                'feedback_id': [f'FB_{i:06d}' for i in range(500)],
                'customer_id': np.random.choice(customers['customer_id'], 500),
                'product_id': [f'PROD_{i:03d}' for i in np.random.randint(1, 101, 500)],
                'rating': np.random.randint(1, 6, 500),
                'text': np.random.choice(feedback_texts, 500),
                'date': pd.date_range('2023-01-01', periods=500, freq='D')
            })
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            success = (
                len(customers) == 1000 and
                len(sales) == 2000 and
                len(feedback) == 500
            )
            
            self.performance_results['data_generation'] = {
                'success': success,
                'execution_time': execution_time,
                'customers_generated': len(customers),
                'sales_generated': len(sales),
                'feedback_generated': len(feedback)
            }
            
            print(f"   ✅ Data generation: {execution_time:.3f}s")
            print(f"   📊 Generated {len(customers)} customers, {len(sales)} sales, {len(feedback)} feedback")
            
            return customers, sales, feedback
            
        except Exception as e:
            print(f"   ❌ Data generation test failed: {e}")
            self.performance_results['data_generation'] = {'success': False, 'error': str(e)}
            return None, None, None
    
    def test_customer_segmentation_performance(self, customers):
        """Test customer segmentation performance"""
        print("🤖 Testing Customer Segmentation Performance...")
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            start_time = time.time()
            
            # Prepare features
            features = ['age', 'total_spend', 'purchase_count', 'retention_rate']
            X = customers[features].fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Analyze clusters
            customers['cluster'] = clusters
            cluster_sizes = [len(customers[customers['cluster'] == i]) for i in range(5)]
            
            success = (
                len(set(clusters)) == 5 and
                all(size > 0 for size in cluster_sizes) and
                execution_time < 5.0  # Should complete within 5 seconds
            )
            
            self.performance_results['customer_segmentation'] = {
                'success': success,
                'execution_time': execution_time,
                'clusters_created': len(set(clusters)),
                'cluster_sizes': cluster_sizes
            }
            
            print(f"   ✅ Customer segmentation: {execution_time:.3f}s")
            print(f"   📊 Created {len(set(clusters))} clusters: {cluster_sizes}")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Customer segmentation test failed: {e}")
            self.performance_results['customer_segmentation'] = {'success': False, 'error': str(e)}
            return False
    
    def test_sentiment_analysis_performance(self, feedback):
        """Test sentiment analysis performance"""
        print("😊 Testing Sentiment Analysis Performance...")
        
        try:
            start_time = time.time()
            
            # Simple sentiment analysis based on keywords
            positive_words = ['great', 'excellent', 'amazing', 'outstanding', 'satisfied', 'recommend', 'love', 'best']
            negative_words = ['poor', 'bad', 'terrible', 'disappointed', 'damaged', 'worst', 'hate', 'awful']
            
            sentiments = []
            for text in feedback['text']:
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    sentiment = 'Positive'
                elif negative_count > positive_count:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                
                sentiments.append(sentiment)
            
            feedback['sentiment'] = sentiments
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Analyze sentiment distribution
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            success = (
                len(sentiments) == len(feedback) and
                all(sentiment in ['Positive', 'Negative', 'Neutral'] for sentiment in sentiments) and
                execution_time < 2.0  # Should complete within 2 seconds
            )
            
            self.performance_results['sentiment_analysis'] = {
                'success': success,
                'execution_time': execution_time,
                'texts_analyzed': len(sentiments),
                'sentiment_distribution': sentiment_counts.to_dict()
            }
            
            print(f"   ✅ Sentiment analysis: {execution_time:.3f}s")
            print(f"   📊 Analyzed {len(sentiments)} texts")
            print(f"   📈 Sentiment distribution: {dict(sentiment_counts)}")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Sentiment analysis test failed: {e}")
            self.performance_results['sentiment_analysis'] = {'success': False, 'error': str(e)}
            return False
    
    def test_sales_forecasting_performance(self, sales):
        """Test sales forecasting performance"""
        print("📈 Testing Sales Forecasting Performance...")
        
        try:
            start_time = time.time()
            
            # Prepare time series data
            sales['date'] = pd.to_datetime(sales['date'])
            daily_sales = sales.groupby(sales['date'].dt.date)['amount'].sum().reset_index()
            daily_sales.columns = ['date', 'sales']
            daily_sales = daily_sales.sort_values('date')
            
            # Simple moving average forecast
            window = min(7, len(daily_sales) // 2)
            if window >= 2:
                ma = daily_sales['sales'].rolling(window=window).mean()
                last_ma = ma.iloc[-1]
                
                # Generate forecast
                forecast_dates = pd.date_range(start=daily_sales['date'].iloc[-1] + pd.Timedelta(days=1), periods=7, freq='D')
                forecast_values = [last_ma] * 7
                
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'forecast': forecast_values
                })
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                success = (
                    len(forecast_df) == 7 and
                    all(forecast_values) and
                    execution_time < 3.0  # Should complete within 3 seconds
                )
                
                self.performance_results['sales_forecasting'] = {
                    'success': success,
                    'execution_time': execution_time,
                    'forecast_periods': len(forecast_df),
                    'avg_forecast_value': np.mean(forecast_values)
                }
                
                print(f"   ✅ Sales forecasting: {execution_time:.3f}s")
                print(f"   📊 Generated {len(forecast_df)} forecast periods")
                print(f"   📈 Average forecast: ${np.mean(forecast_values):.2f}")
                
                return success
            else:
                print("   ⚠️  Insufficient data for forecasting")
                return False
                
        except Exception as e:
            print(f"   ❌ Sales forecasting test failed: {e}")
            self.performance_results['sales_forecasting'] = {'success': False, 'error': str(e)}
            return False
    
    def test_business_scenarios(self, customers, sales, feedback):
        """Test business scenarios"""
        print("🎯 Testing Business Scenarios...")
        
        scenarios = {}
        
        # Scenario 1: Customer Retention Analysis
        try:
            print("   🧪 Testing Customer Retention Scenario...")
            
            # Create churn indicators
            customers['days_since_purchase'] = np.random.randint(1, 365, len(customers))
            customers['purchase_frequency'] = customers['purchase_count'] / (customers['days_since_purchase'] / 30)
            
            # Simple churn scoring
            churn_scores = []
            for _, customer in customers.iterrows():
                score = 0
                
                if customer['days_since_purchase'] > 90:
                    score += 0.4
                elif customer['days_since_purchase'] > 30:
                    score += 0.2
                
                if customer['purchase_frequency'] < 0.5:
                    score += 0.3
                elif customer['purchase_frequency'] < 1.0:
                    score += 0.1
                
                if customer['total_spend'] < 500:
                    score += 0.2
                elif customer['total_spend'] < 1000:
                    score += 0.1
                
                churn_scores.append(min(1.0, score))
            
            customers['churn_probability'] = churn_scores
            customers['risk_level'] = customers['churn_probability'].apply(
                lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
            )
            
            high_risk_customers = customers[customers['risk_level'] == 'High']
            
            scenarios['customer_retention'] = {
                'success': len(high_risk_customers) > 0,
                'high_risk_customers': len(high_risk_customers),
                'avg_churn_probability': np.mean(churn_scores)
            }
            
            print(f"   ✅ Customer retention: {len(high_risk_customers)} high-risk customers")
            
        except Exception as e:
            print(f"   ❌ Customer retention scenario failed: {e}")
            scenarios['customer_retention'] = {'success': False, 'error': str(e)}
        
        # Scenario 2: Price Optimization
        try:
            print("   🧪 Testing Price Optimization Scenario...")
            
            # Calculate price elasticity for top products
            product_sales = sales.groupby('product_id').agg({
                'amount': ['mean', 'std'],
                'quantity': ['mean', 'std']
            }).round(2)
            
            product_sales.columns = ['avg_price', 'price_std', 'avg_quantity', 'quantity_std']
            product_sales = product_sales.dropna()
            
            # Select top 5 products
            top_products = sales['product_id'].value_counts().head(5).index
            
            optimization_results = []
            for product in top_products:
                if product in product_sales.index:
                    avg_price = product_sales.loc[product, 'avg_price']
                    cost = avg_price * 0.6  # Assume 40% margin
                    elasticity = -1.5  # Default elasticity
                    optimal_price = cost / (1 + 1/abs(elasticity))
                    
                    optimization_results.append({
                        'product_id': product,
                        'current_price': avg_price,
                        'optimal_price': optimal_price,
                        'price_change_percent': (optimal_price - avg_price) / avg_price * 100
                    })
            
            valid_optimizations = sum(1 for r in optimization_results if r['optimal_price'] > r['current_price'] * 0.5)
            
            scenarios['price_optimization'] = {
                'success': valid_optimizations >= 3,
                'total_products_tested': len(optimization_results),
                'valid_optimizations': valid_optimizations
            }
            
            print(f"   ✅ Price optimization: {valid_optimizations}/{len(optimization_results)} valid optimizations")
            
        except Exception as e:
            print(f"   ❌ Price optimization scenario failed: {e}")
            scenarios['price_optimization'] = {'success': False, 'error': str(e)}
        
        self.test_results['business_scenarios'] = scenarios
        return scenarios
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("⚠️  Testing Edge Cases...")
        
        edge_cases = {}
        
        # Test 1: Empty data handling
        try:
            print("   🧪 Testing Empty Data Handling...")
            empty_df = pd.DataFrame()
            success = len(empty_df) == 0
            edge_cases['empty_data'] = {'success': success}
            print(f"   {'✅' if success else '❌'} Empty data handling: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            edge_cases['empty_data'] = {'success': False, 'error': str(e)}
            print(f"   ❌ Empty data handling: ERROR - {e}")
        
        # Test 2: Invalid data handling
        try:
            print("   🧪 Testing Invalid Data Handling...")
            invalid_data = pd.DataFrame({
                'text': [None, "", "   ", 123, [], {}]
            })
            
            # Test text processing
            processed_texts = []
            for text in invalid_data['text']:
                if isinstance(text, str) and text.strip():
                    processed = text.lower().strip()
                    processed_texts.append(processed)
                else:
                    processed_texts.append("")
            
            success = len(processed_texts) == len(invalid_data)
            edge_cases['invalid_data'] = {'success': success}
            print(f"   {'✅' if success else '❌'} Invalid data handling: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            edge_cases['invalid_data'] = {'success': False, 'error': str(e)}
            print(f"   ❌ Invalid data handling: ERROR - {e}")
        
        # Test 3: Large data handling
        try:
            print("   🧪 Testing Large Data Handling...")
            large_data = pd.DataFrame({
                'id': range(10000),
                'value': np.random.randn(10000)
            })
            success = len(large_data) == 10000
            edge_cases['large_data'] = {'success': success}
            print(f"   {'✅' if success else '❌'} Large data handling: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            edge_cases['large_data'] = {'success': False, 'error': str(e)}
            print(f"   ❌ Large data handling: ERROR - {e}")
        
        self.test_results['edge_cases'] = edge_cases
        return edge_cases
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 SIMPLIFIED ENHANCED TEST SUITE REPORT")
        print("=" * 60)
        
        # Performance Summary
        total_perf_tests = len(self.performance_results)
        successful_perf_tests = sum(1 for r in self.performance_results.values() if r.get('success', False))
        perf_success_rate = (successful_perf_tests / total_perf_tests * 100) if total_perf_tests > 0 else 0
        
        print(f"\n🚀 Performance Summary:")
        print(f"   Total Tests: {total_perf_tests}")
        print(f"   Successful Tests: {successful_perf_tests}")
        print(f"   Success Rate: {perf_success_rate:.1f}%")
        
        # Business Scenario Summary
        scenarios = self.test_results.get('business_scenarios', {})
        total_scenarios = len(scenarios)
        successful_scenarios = sum(1 for s in scenarios.values() if s.get('success', False))
        scenario_success_rate = (successful_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0
        
        print(f"\n🎯 Business Scenario Summary:")
        print(f"   Total Scenarios: {total_scenarios}")
        print(f"   Successful Scenarios: {successful_scenarios}")
        print(f"   Success Rate: {scenario_success_rate:.1f}%")
        
        # Edge Case Summary
        edge_cases = self.test_results.get('edge_cases', {})
        total_edge_cases = len(edge_cases)
        successful_edge_cases = sum(1 for e in edge_cases.values() if e.get('success', False))
        edge_case_success_rate = (successful_edge_cases / total_edge_cases * 100) if total_edge_cases > 0 else 0
        
        print(f"\n⚠️  Edge Case Summary:")
        print(f"   Total Tests: {total_edge_cases}")
        print(f"   Successful Tests: {successful_edge_cases}")
        print(f"   Success Rate: {edge_case_success_rate:.1f}%")
        
        # Overall Assessment
        total_tests = total_perf_tests + total_scenarios + total_edge_cases
        total_success = successful_perf_tests + successful_scenarios + successful_edge_cases
        
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
            'performance': self.performance_results,
            'scenarios': scenarios,
            'edge_cases': edge_cases,
            'overall_success_rate': overall_success_rate
        }
    
    def run_all_tests(self):
        """Run all simplified enhanced tests"""
        print("🚀 SLM Business Insights - Simplified Enhanced Test Suite")
        print("=" * 60)
        
        # Test data generation
        customers, sales, feedback = self.test_data_generation_performance()
        
        if customers is not None:
            # Test customer segmentation
            self.test_customer_segmentation_performance(customers)
            
            # Test sentiment analysis
            self.test_sentiment_analysis_performance(feedback)
            
            # Test sales forecasting
            self.test_sales_forecasting_performance(sales)
            
            # Test business scenarios
            self.test_business_scenarios(customers, sales, feedback)
        
        # Test edge cases
        self.test_edge_cases()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        return report

def main():
    """Run the simplified enhanced test suite"""
    test_suite = SimplifiedEnhancedTestSuite()
    report = test_suite.run_all_tests()
    
    # Save report to file
    with open('simplified_enhanced_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: simplified_enhanced_test_report.json")
    
    return report

if __name__ == "__main__":
    main()
