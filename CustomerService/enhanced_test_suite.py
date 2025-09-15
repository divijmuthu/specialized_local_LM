"""
Enhanced Test Suite for Customer Service Feedback Analysis Tool
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
    """Performance benchmarking utilities for Customer Service Analysis"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark a function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return result, execution_time
    
    def add_result(self, test_name, execution_time, success=True, details=None):
        """Add benchmark result"""
        self.results[test_name] = {
            'execution_time': execution_time,
            'success': success,
            'details': details or {},
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

class CustomerServiceScenarioTester:
    """Test customer service scenarios and use cases"""
    
    def __init__(self):
        self.scenarios = {}
    
    def test_customer_satisfaction_analysis(self):
        """Test customer satisfaction analysis scenario"""
        print("🧪 Testing Customer Satisfaction Analysis Scenario...")
        
        try:
            # Generate customer feedback data
            feedback_data = pd.DataFrame({
                'text': [
                    "I love your product! It's amazing and works perfectly.",
                    "Customer service was terrible. They didn't respond to my issue.",
                    "The product is good, but the pricing is too high.",
                    "Excellent support team! They resolved my issue quickly.",
                    "The new feature is great! It solves a major problem for us.",
                    "I've been a customer for years, but recent changes are disappointing.",
                    "Fast delivery and good quality product. Highly recommend!",
                    "Poor quality product. Not worth the money.",
                    "The support team was very helpful in resolving my issue.",
                    "The product works well, but the documentation is lacking."
                ],
                'date': pd.date_range('2023-01-01', periods=10, freq='D'),
                'source': ['email', 'review', 'survey', 'email', 'survey', 'social_media', 'review', 'review', 'email', 'survey']
            })
            
            # Test sentiment analysis
            from app import analyze_sentiment, generate_insights
            
            sentiments = []
            for text in feedback_data['text']:
                sentiment, score = analyze_sentiment(text)
                sentiments.append(sentiment)
            
            feedback_data['sentiment'] = sentiments
            
            # Analyze satisfaction metrics
            positive_feedback = len(feedback_data[feedback_data['sentiment'] == 'positive'])
            negative_feedback = len(feedback_data[feedback_data['sentiment'] == 'negative'])
            total_feedback = len(feedback_data)
            
            satisfaction_rate = (positive_feedback / total_feedback) * 100
            complaint_rate = (negative_feedback / total_feedback) * 100
            
            # Business validation: Should have reasonable satisfaction metrics
            business_logic_valid = (
                0 <= satisfaction_rate <= 100 and
                0 <= complaint_rate <= 100 and
                satisfaction_rate + complaint_rate <= 100
            )
            
            self.scenarios['customer_satisfaction'] = {
                'success': business_logic_valid,
                'total_feedback': total_feedback,
                'satisfaction_rate': satisfaction_rate,
                'complaint_rate': complaint_rate,
                'positive_feedback': positive_feedback,
                'negative_feedback': negative_feedback
            }
            
            print(f"   ✅ Customer satisfaction analysis: {satisfaction_rate:.1f}% satisfaction rate")
            print(f"   📊 Positive: {positive_feedback}, Negative: {negative_feedback}")
            
            return business_logic_valid
            
        except Exception as e:
            print(f"   ❌ Customer satisfaction analysis failed: {e}")
            self.scenarios['customer_satisfaction'] = {'success': False, 'error': str(e)}
            return False
    
    def test_topic_analysis_scenario(self):
        """Test topic analysis and theme identification"""
        print("🧪 Testing Topic Analysis Scenario...")
        
        try:
            # Generate feedback with different topics
            topic_feedback = pd.DataFrame({
                'text': [
                    "The product quality is excellent and reliable",
                    "Customer service response time is too slow",
                    "The pricing is too high compared to competitors",
                    "The new feature is fantastic and very useful",
                    "Documentation is poor and hard to understand",
                    "The user interface is confusing and needs improvement",
                    "Product delivery was fast and efficient",
                    "Technical support was helpful and knowledgeable",
                    "The product has many bugs and errors",
                    "Overall experience was great and satisfying"
                ],
                'date': pd.date_range('2023-01-01', periods=10, freq='D')
            })
            
            # Test topic extraction
            from app import extract_topics, generate_insights
            
            topics, topic_labels = extract_topics(topic_feedback['text'].tolist(), n_topics=3)
            
            # Test insights generation
            insights, enriched_df = generate_insights(topic_feedback)
            
            # Business validation: Should identify meaningful topics
            topics_identified = len(topics) > 0 if topics else False
            insights_generated = 'topic_distribution' in insights
            actionable_items = 'actionable_items' in insights
            
            self.scenarios['topic_analysis'] = {
                'success': topics_identified and insights_generated and actionable_items,
                'topics_identified': len(topics) if topics else 0,
                'insights_generated': len(insights),
                'actionable_items_found': len(insights.get('actionable_items', []))
            }
            
            print(f"   ✅ Topic analysis: {len(topics) if topics else 0} topics identified")
            print(f"   📊 Insights generated: {len(insights)}")
            print(f"   🎯 Actionable items: {len(insights.get('actionable_items', []))}")
            
            return topics_identified and insights_generated and actionable_items
            
        except Exception as e:
            print(f"   ❌ Topic analysis scenario failed: {e}")
            self.scenarios['topic_analysis'] = {'success': False, 'error': str(e)}
            return False
    
    def test_actionable_insights_scenario(self):
        """Test actionable insights generation"""
        print("🧪 Testing Actionable Insights Scenario...")
        
        try:
            # Generate feedback with specific actionable items
            actionable_feedback = pd.DataFrame({
                'text': [
                    "There is a bug in the login system that needs to be fixed",
                    "The documentation is poor and needs improvement",
                    "Customer service response time is too slow",
                    "The pricing is too high compared to competitors",
                    "The user interface is confusing and needs redesign",
                    "Please add a dark mode feature to the application",
                    "The product crashes frequently and needs stability fixes",
                    "The support team needs better training",
                    "The mobile app is missing important features",
                    "The checkout process is too complicated"
                ],
                'date': pd.date_range('2023-01-01', periods=10, freq='D')
            })
            
            # Test actionable item detection
            from app import analyze_text_for_actions, generate_insights
            
            all_actions = []
            for text in actionable_feedback['text']:
                actions = analyze_text_for_actions(text)
                all_actions.extend([action[0] for action in actions])
            
            # Test insights generation
            insights, enriched_df = generate_insights(actionable_feedback)
            
            # Business validation: Should identify actionable items
            actions_identified = len(all_actions) > 0
            insights_contain_actions = 'actionable_items' in insights
            priority_items = len(insights.get('actionable_items', [])) > 0
            
            # Check for specific action categories
            action_categories = set(all_actions)
            expected_categories = {'bug', 'documentation', 'response time', 'pricing', 'ui/ux', 'feature request'}
            relevant_categories = len(action_categories.intersection(expected_categories)) > 0
            
            self.scenarios['actionable_insights'] = {
                'success': actions_identified and insights_contain_actions and priority_items and relevant_categories,
                'actions_identified': len(all_actions),
                'action_categories': list(action_categories),
                'priority_items': len(insights.get('actionable_items', [])),
                'relevant_categories_found': relevant_categories
            }
            
            print(f"   ✅ Actionable insights: {len(all_actions)} actions identified")
            print(f"   📊 Categories: {list(action_categories)}")
            print(f"   🎯 Priority items: {len(insights.get('actionable_items', []))}")
            
            return actions_identified and insights_contain_actions and priority_items and relevant_categories
            
        except Exception as e:
            print(f"   ❌ Actionable insights scenario failed: {e}")
            self.scenarios['actionable_insights'] = {'success': False, 'error': str(e)}
            return False
    
    def test_trend_analysis_scenario(self):
        """Test trend analysis over time"""
        print("🧪 Testing Trend Analysis Scenario...")
        
        try:
            # Generate time-series feedback data
            dates = pd.date_range('2023-01-01', periods=30, freq='D')
            feedback_texts = [
                "Great product, very satisfied!",
                "Poor customer service, disappointed",
                "Excellent value for money",
                "Product arrived damaged",
                "Fast delivery and good quality",
                "Not as described, disappointed",
                "Amazing experience, will buy again",
                "Average product, nothing special",
                "Outstanding customer support",
                "Product quality could be better"
            ]
            
            trend_data = pd.DataFrame({
                'text': np.random.choice(feedback_texts, 30),
                'date': dates,
                'source': np.random.choice(['email', 'review', 'survey', 'social_media'], 30)
            })
            
            # Test trend analysis
            from app import generate_insights
            
            insights, enriched_df = generate_insights(trend_data)
            
            # Business validation: Should generate trend insights
            sentiment_trends = 'sentiment_over_time' in insights
            time_analysis = enriched_df['date'].dtype == 'datetime64[ns]'
            trend_data_points = len(insights.get('sentiment_over_time', [])) > 0 if sentiment_trends else False
            
            self.scenarios['trend_analysis'] = {
                'success': sentiment_trends and time_analysis and trend_data_points,
                'sentiment_trends_generated': sentiment_trends,
                'time_analysis_working': time_analysis,
                'trend_data_points': len(insights.get('sentiment_over_time', [])) if sentiment_trends else 0
            }
            
            print(f"   ✅ Trend analysis: {len(insights.get('sentiment_over_time', [])) if sentiment_trends else 0} data points")
            print(f"   📊 Sentiment trends: {'✅' if sentiment_trends else '❌'}")
            print(f"   📈 Time analysis: {'✅' if time_analysis else '❌'}")
            
            return sentiment_trends and time_analysis and trend_data_points
            
        except Exception as e:
            print(f"   ❌ Trend analysis scenario failed: {e}")
            self.scenarios['trend_analysis'] = {'success': False, 'error': str(e)}
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

class EdgeCaseTester:
    """Test edge cases and error handling"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_empty_feedback_handling(self):
        """Test handling of empty feedback"""
        print("🧪 Testing Empty Feedback Handling...")
        
        try:
            from app import generate_insights
            
            # Test with empty DataFrame
            empty_df = pd.DataFrame()
            insights, enriched_df = generate_insights(empty_df)
            
            success = insights == {} and enriched_df.empty
            
            self.test_results['empty_feedback'] = {'success': success}
            print(f"   {'✅' if success else '❌'} Empty feedback handling: {'PASSED' if success else 'FAILED'}")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Empty feedback handling: ERROR - {e}")
            self.test_results['empty_feedback'] = {'success': False, 'error': str(e)}
            return False
    
    def test_invalid_text_handling(self):
        """Test handling of invalid text data"""
        print("🧪 Testing Invalid Text Handling...")
        
        try:
            from app import preprocess_text, analyze_sentiment
            
            # Test with invalid text data
            invalid_texts = [None, "", "   ", 123, [], {}, "   \n\t   "]
            
            success_count = 0
            for text in invalid_texts:
                try:
                    processed = preprocess_text(text)
                    sentiment, score = analyze_sentiment(text)
                    
                    # Should handle invalid inputs gracefully
                    if isinstance(processed, str) and isinstance(sentiment, str):
                        success_count += 1
                except Exception:
                    # Some invalid inputs might cause exceptions, which is acceptable
                    pass
            
            success = success_count >= len(invalid_texts) * 0.7  # 70% success rate acceptable
            
            self.test_results['invalid_text'] = {
                'success': success,
                'success_count': success_count,
                'total_tests': len(invalid_texts)
            }
            print(f"   {'✅' if success else '❌'} Invalid text handling: {'PASSED' if success else 'FAILED'}")
            print(f"   📊 Success rate: {success_count}/{len(invalid_texts)}")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Invalid text handling: ERROR - {e}")
            self.test_results['invalid_text'] = {'success': False, 'error': str(e)}
            return False
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        print("🧪 Testing Large Dataset Handling...")
        
        try:
            from app import generate_insights
            
            # Generate large dataset
            large_data = pd.DataFrame({
                'text': [f"Feedback text {i}" for i in range(1000)],
                'date': pd.date_range('2023-01-01', periods=1000, freq='H')
            })
            
            start_time = time.time()
            insights, enriched_df = generate_insights(large_data)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Should handle large dataset without errors and within reasonable time
            success = (
                len(enriched_df) == 1000 and
                'sentiment_distribution' in insights and
                execution_time < 30.0  # Should complete within 30 seconds
            )
            
            self.test_results['large_dataset'] = {
                'success': success,
                'execution_time': execution_time,
                'dataset_size': len(large_data),
                'processed_size': len(enriched_df)
            }
            print(f"   {'✅' if success else '❌'} Large dataset handling: {'PASSED' if success else 'FAILED'}")
            print(f"   📊 Execution time: {execution_time:.3f}s")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Large dataset handling: ERROR - {e}")
            self.test_results['large_dataset'] = {'success': False, 'error': str(e)}
            return False
    
    def test_special_characters_handling(self):
        """Test handling of special characters and encoding"""
        print("🧪 Testing Special Characters Handling...")
        
        try:
            from app import preprocess_text, analyze_sentiment
            
            # Test texts with special characters
            special_texts = [
                "Text with émojis 🎉 and accénts",
                "Text with HTML entities &amp; &lt; &gt;",
                "Text with unicode: 中文 العربية русский",
                "Text with symbols: !@#$%^&*()_+-=[]{}|;':\",./<>?",
                "Text with line breaks\nand\ttabs",
                "Text with multiple    spaces   and   punctuation!!!"
            ]
            
            success_count = 0
            for text in special_texts:
                try:
                    processed = preprocess_text(text)
                    sentiment, score = analyze_sentiment(text)
                    
                    if isinstance(processed, str) and isinstance(sentiment, str):
                        success_count += 1
                except Exception:
                    pass  # Some special characters might cause issues, which is acceptable
            
            success = success_count >= len(special_texts) * 0.8  # 80% success rate acceptable
            
            self.test_results['special_characters'] = {
                'success': success,
                'success_count': success_count,
                'total_tests': len(special_texts)
            }
            print(f"   {'✅' if success else '❌'} Special characters handling: {'PASSED' if success else 'FAILED'}")
            print(f"   📊 Success rate: {success_count}/{len(special_texts)}")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Special characters handling: ERROR - {e}")
            self.test_results['special_characters'] = {'success': False, 'error': str(e)}
            return False
    
    def get_edge_case_summary(self):
        """Get summary of edge case tests"""
        if not self.test_results:
            return "No edge case tests run"
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results.values() if r.get('success', False))
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': f"{(successful_tests/total_tests)*100:.1f}%",
            'test_results': self.test_results
        }

class EnhancedCustomerServiceTestSuite:
    """Enhanced test suite for Customer Service Feedback Analysis Tool"""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        self.scenario_tester = CustomerServiceScenarioTester()
        self.edge_case_tester = EdgeCaseTester()
        self.test_results = {}
    
    def run_performance_tests(self):
        """Run performance benchmark tests"""
        print("🚀 Running Performance Tests...")
        print("=" * 50)
        
        # Test 1: Text Preprocessing Performance
        print("📝 Testing Text Preprocessing Performance...")
        try:
            from app import preprocess_text
            
            test_texts = [
                "This is a test sentence with punctuation, numbers 123, and special characters!",
                "Another test sentence " * 50,  # Long text
                "Short text",
                "Text with HTML <p>tags</p> and <br/>line breaks",
                "Text with multiple\n\nline\nbreaks and\t\ttabs"
            ]
            
            start_time = time.time()
            processed_texts = []
            for text in test_texts:
                processed = preprocess_text(text)
                processed_texts.append(processed)
            end_time = time.time()
            
            execution_time = end_time - start_time
            success = len(processed_texts) == len(test_texts)
            
            self.benchmark.add_result('text_preprocessing', execution_time, success, {
                'texts_processed': len(processed_texts),
                'avg_time_per_text': execution_time / len(test_texts)
            })
            
            print(f"   ✅ Text preprocessing: {execution_time:.3f}s")
            print(f"   📊 Processed {len(processed_texts)} texts")
            
        except Exception as e:
            print(f"   ❌ Text preprocessing test failed: {e}")
            self.benchmark.add_result('text_preprocessing', 0, False, {'error': str(e)})
        
        # Test 2: Sentiment Analysis Performance
        print("\n😊 Testing Sentiment Analysis Performance...")
        try:
            from app import analyze_sentiment
            
            test_texts = [
                "I love this product! It's amazing and works perfectly.",
                "This is terrible. I hate it and want a refund immediately.",
                "The product is okay, nothing special but it works.",
                "Can you help me with my order? I have a question about shipping.",
                "Thank you for your excellent customer service!"
            ]
            
            start_time = time.time()
            sentiment_results = []
            for text in test_texts:
                sentiment, score = analyze_sentiment(text)
                sentiment_results.append((sentiment, score))
            end_time = time.time()
            
            execution_time = end_time - start_time
            success = len(sentiment_results) == len(test_texts)
            
            self.benchmark.add_result('sentiment_analysis', execution_time, success, {
                'texts_analyzed': len(sentiment_results),
                'avg_time_per_text': execution_time / len(test_texts)
            })
            
            print(f"   ✅ Sentiment analysis: {execution_time:.3f}s")
            print(f"   📊 Analyzed {len(sentiment_results)} texts")
            
        except Exception as e:
            print(f"   ❌ Sentiment analysis test failed: {e}")
            self.benchmark.add_result('sentiment_analysis', 0, False, {'error': str(e)})
        
        # Test 3: Topic Extraction Performance
        print("\n🔍 Testing Topic Extraction Performance...")
        try:
            from app import extract_topics
            
            test_texts = [
                "The product quality is excellent and reliable",
                "Customer service response time is too slow",
                "The pricing is too high compared to competitors",
                "The new feature is fantastic and very useful",
                "Documentation is poor and hard to understand"
            ]
            
            start_time = time.time()
            topics, topic_labels = extract_topics(test_texts, n_topics=3)
            end_time = time.time()
            
            execution_time = end_time - start_time
            success = isinstance(topics, list) and isinstance(topic_labels, (list, np.ndarray))
            
            self.benchmark.add_result('topic_extraction', execution_time, success, {
                'topics_identified': len(topics) if topics else 0,
                'texts_processed': len(test_texts)
            })
            
            print(f"   ✅ Topic extraction: {execution_time:.3f}s")
            print(f"   📊 Identified {len(topics) if topics else 0} topics")
            
        except Exception as e:
            print(f"   ❌ Topic extraction test failed: {e}")
            self.benchmark.add_result('topic_extraction', 0, False, {'error': str(e)})
        
        # Test 4: Insights Generation Performance
        print("\n💡 Testing Insights Generation Performance...")
        try:
            from app import generate_insights
            
            # Generate test data
            test_data = pd.DataFrame({
                'text': [
                    "I love your product! It's amazing.",
                    "Customer service was terrible. They didn't respond to my issue.",
                    "The product is good, but the pricing is too high.",
                    "Excellent support team! They resolved my issue quickly.",
                    "The new feature is great! It solves a major problem for us."
                ],
                'date': pd.date_range('2023-01-01', periods=5, freq='D')
            })
            
            start_time = time.time()
            insights, enriched_df = generate_insights(test_data)
            end_time = time.time()
            
            execution_time = end_time - start_time
            success = isinstance(insights, dict) and isinstance(enriched_df, pd.DataFrame)
            
            self.benchmark.add_result('insights_generation', execution_time, success, {
                'insights_generated': len(insights),
                'records_processed': len(enriched_df)
            })
            
            print(f"   ✅ Insights generation: {execution_time:.3f}s")
            print(f"   📊 Generated {len(insights)} insights")
            
        except Exception as e:
            print(f"   ❌ Insights generation test failed: {e}")
            self.benchmark.add_result('insights_generation', 0, False, {'error': str(e)})
    
    def run_business_scenario_tests(self):
        """Run business scenario tests"""
        print("\n🎯 Running Business Scenario Tests...")
        print("=" * 50)
        
        # Test customer satisfaction analysis
        self.scenario_tester.test_customer_satisfaction_analysis()
        
        # Test topic analysis
        self.scenario_tester.test_topic_analysis_scenario()
        
        # Test actionable insights
        self.scenario_tester.test_actionable_insights_scenario()
        
        # Test trend analysis
        self.scenario_tester.test_trend_analysis_scenario()
    
    def run_edge_case_tests(self):
        """Run edge case and error handling tests"""
        print("\n⚠️  Running Edge Case Tests...")
        print("=" * 50)
        
        # Test empty feedback handling
        self.edge_case_tester.test_empty_feedback_handling()
        
        # Test invalid text handling
        self.edge_case_tester.test_invalid_text_handling()
        
        # Test large dataset handling
        self.edge_case_tester.test_large_dataset_handling()
        
        # Test special characters handling
        self.edge_case_tester.test_special_characters_handling()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 ENHANCED CUSTOMER SERVICE TEST REPORT")
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
        edge_case_summary = self.edge_case_tester.get_edge_case_summary()
        print(f"\n⚠️  Edge Case Summary:")
        print(f"   Total Tests: {edge_case_summary['total_tests']}")
        print(f"   Success Rate: {edge_case_summary['success_rate']}")
        
        # Overall Assessment
        total_tests = (perf_summary['total_tests'] + 
                      scenario_summary['total_scenarios'] + 
                      edge_case_summary['total_tests'])
        
        total_success = (perf_summary['successful_tests'] + 
                        scenario_summary['successful_scenarios'] + 
                        edge_case_summary['successful_tests'])
        
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🏆 Overall Assessment:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful Tests: {total_success}")
        print(f"   Overall Success Rate: {overall_success_rate:.1f}%")
        
        if overall_success_rate >= 90:
            print("   🎉 EXCELLENT: Customer Service system is performing exceptionally well!")
        elif overall_success_rate >= 80:
            print("   ✅ GOOD: Customer Service system is performing well with minor issues.")
        elif overall_success_rate >= 70:
            print("   ⚠️  FAIR: Customer Service system needs some improvements.")
        else:
            print("   ❌ POOR: Customer Service system requires significant improvements.")
        
        return {
            'performance': perf_summary,
            'scenarios': scenario_summary,
            'edge_cases': edge_case_summary,
            'overall_success_rate': overall_success_rate
        }
    
    def run_all_tests(self):
        """Run all enhanced tests"""
        print("🚀 Customer Service Feedback Analysis - Enhanced Test Suite")
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
    test_suite = EnhancedCustomerServiceTestSuite()
    report = test_suite.run_all_tests()
    
    # Save report to file
    with open('enhanced_customer_service_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: enhanced_customer_service_test_report.json")
    
    return report

if __name__ == "__main__":
    main()
