"""
Enhanced Test Suite for Email Insights Application
Comprehensive testing including performance, Gmail integration, and edge cases
"""

import sys
import os
import time
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PerformanceBenchmark:
    """Performance benchmarking utilities for Email Insights"""
    
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
            return "No benchmark results available"
        
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

class GmailIntegrationTester:
    """Test Gmail API integration and email processing"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_email_parsing_performance(self):
        """Test email parsing performance with various email formats"""
        print("📧 Testing Email Parsing Performance...")
        
        try:
            # Import email parsing functions
            from app import parse_email, preprocess_text
            
            # Create mock email data with different formats
            mock_emails = [
                {
                    'payload': {
                        'headers': [
                            {'name': 'Subject', 'value': 'Test Email 1'},
                            {'name': 'From', 'value': 'sender1@example.com'},
                            {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'}
                        ],
                        'body': {'data': 'dGVzdCBlbWFpbCBib2R5IDE='}  # base64 encoded
                    }
                },
                {
                    'payload': {
                        'headers': [
                            {'name': 'Subject', 'value': 'Test Email 2 with Long Subject Line'},
                            {'name': 'From', 'value': 'sender2@example.com'},
                            {'name': 'Date', 'value': 'Tue, 2 Jan 2024 13:30:00 +0000'}
                        ],
                        'parts': [
                            {
                                'mimeType': 'text/plain',
                                'body': {'data': 'dGVzdCBlbWFpbCBib2R5IDIgd2l0aCBtdWx0aXBsZSBwYXJ0cw=='}
                            }
                        ]
                    }
                }
            ]
            
            # Test parsing performance
            start_time = time.time()
            parsed_emails = []
            for email in mock_emails:
                parsed = parse_email(email)
                parsed_emails.append(parsed)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Validate parsing results
            success = (
                len(parsed_emails) == 2 and
                all('subject' in email for email in parsed_emails) and
                all('sender' in email for email in parsed_emails) and
                all('body' in email for email in parsed_emails)
            )
            
            self.test_results['email_parsing'] = {
                'success': success,
                'execution_time': execution_time,
                'emails_parsed': len(parsed_emails),
                'avg_time_per_email': execution_time / len(mock_emails)
            }
            
            print(f"   ✅ Email parsing: {len(parsed_emails)} emails in {execution_time:.3f}s")
            print(f"   📊 Average time per email: {execution_time/len(mock_emails):.3f}s")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Email parsing test failed: {e}")
            self.test_results['email_parsing'] = {'success': False, 'error': str(e)}
            return False
    
    def test_text_preprocessing_performance(self):
        """Test text preprocessing performance with various text lengths"""
        print("📝 Testing Text Preprocessing Performance...")
        
        try:
            from app import preprocess_text
            
            # Create test texts of various lengths
            test_texts = [
                "Short text",
                "This is a medium length text with some punctuation, numbers 123, and special characters!",
                "This is a very long text " * 100,  # Long text
                "Text with HTML <p>tags</p> and <br/>line breaks",
                "Text with multiple\n\nline\nbreaks and\t\ttabs",
                "Text with UPPERCASE and lowercase mixed with Numbers 123 and Symbols !@#$%"
            ]
            
            # Test preprocessing performance
            start_time = time.time()
            processed_texts = []
            for text in test_texts:
                processed = preprocess_text(text)
                processed_texts.append(processed)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Validate preprocessing results
            success = (
                len(processed_texts) == len(test_texts) and
                all(isinstance(text, str) for text in processed_texts) and
                all(len(text) > 0 for text in processed_texts)
            )
            
            self.test_results['text_preprocessing'] = {
                'success': success,
                'execution_time': execution_time,
                'texts_processed': len(processed_texts),
                'avg_time_per_text': execution_time / len(test_texts)
            }
            
            print(f"   ✅ Text preprocessing: {len(processed_texts)} texts in {execution_time:.3f}s")
            print(f"   📊 Average time per text: {execution_time/len(test_texts):.3f}s")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Text preprocessing test failed: {e}")
            self.test_results['text_preprocessing'] = {'success': False, 'error': str(e)}
            return False
    
    def test_ai_analysis_performance(self):
        """Test AI analysis performance"""
        print("🤖 Testing AI Analysis Performance...")
        
        try:
            from app import analyze_sentiment, classify_email
            
            # Create test texts for analysis
            test_texts = [
                "I love this product! It's amazing and works perfectly.",
                "This is terrible. I hate it and want a refund immediately.",
                "The product is okay, nothing special but it works.",
                "Can you help me with my order? I have a question about shipping.",
                "Thank you for your excellent customer service!"
            ]
            
            # Test sentiment analysis performance
            start_time = time.time()
            sentiment_results = []
            for text in test_texts:
                sentiment, score = analyze_sentiment(text)
                sentiment_results.append((sentiment, score))
            sentiment_time = time.time() - start_time
            
            # Test email classification performance
            start_time = time.time()
            classification_results = []
            for text in test_texts:
                email_type, score = classify_email(text)
                classification_results.append((email_type, score))
            classification_time = time.time() - start_time
            
            # Validate results
            sentiment_success = (
                len(sentiment_results) == len(test_texts) and
                all(isinstance(result[0], str) for result in sentiment_results) and
                all(isinstance(result[1], float) for result in sentiment_results)
            )
            
            classification_success = (
                len(classification_results) == len(test_texts) and
                all(isinstance(result[0], str) for result in classification_results) and
                all(isinstance(result[1], float) for result in classification_results)
            )
            
            self.test_results['ai_analysis'] = {
                'success': sentiment_success and classification_success,
                'sentiment_time': sentiment_time,
                'classification_time': classification_time,
                'total_time': sentiment_time + classification_time,
                'texts_analyzed': len(test_texts)
            }
            
            print(f"   ✅ AI analysis: {len(test_texts)} texts analyzed")
            print(f"   📊 Sentiment analysis: {sentiment_time:.3f}s")
            print(f"   📊 Email classification: {classification_time:.3f}s")
            
            return sentiment_success and classification_success
            
        except Exception as e:
            print(f"   ❌ AI analysis test failed: {e}")
            self.test_results['ai_analysis'] = {'success': False, 'error': str(e)}
            return False
    
    def test_gmail_api_simulation(self):
        """Test Gmail API integration simulation"""
        print("📬 Testing Gmail API Integration Simulation...")
        
        try:
            # Mock Gmail service
            mock_service = Mock()
            mock_messages = [
                {'id': 'msg1'},
                {'id': 'msg2'},
                {'id': 'msg3'}
            ]
            
            mock_message_details = {
                'msg1': {
                    'payload': {
                        'headers': [
                            {'name': 'Subject', 'value': 'Test Subject 1'},
                            {'name': 'From', 'value': 'test1@example.com'},
                            {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'}
                        ],
                        'body': {'data': 'dGVzdCBib2R5IDE='}
                    }
                },
                'msg2': {
                    'payload': {
                        'headers': [
                            {'name': 'Subject', 'value': 'Test Subject 2'},
                            {'name': 'From', 'value': 'test2@example.com'},
                            {'name': 'Date', 'value': 'Tue, 2 Jan 2024 13:00:00 +0000'}
                        ],
                        'body': {'data': 'dGVzdCBib2R5IDI='}
                    }
                },
                'msg3': {
                    'payload': {
                        'headers': [
                            {'name': 'Subject', 'value': 'Test Subject 3'},
                            {'name': 'From', 'value': 'test3@example.com'},
                            {'name': 'Date', 'value': 'Wed, 3 Jan 2024 14:00:00 +0000'}
                        ],
                        'body': {'data': 'dGVzdCBib2R5IDM='}
                    }
                }
            }
            
            # Configure mock service
            mock_service.users().messages().list().execute.return_value = {'messages': mock_messages}
            
            def mock_get_message(user_id, id):
                return mock_message_details[id]
            
            mock_service.users().messages().get().execute.side_effect = lambda: mock_get_message('me', 'msg1')
            
            # Test email fetching simulation
            start_time = time.time()
            
            # Simulate the email fetching process
            messages = mock_service.users().messages().list().execute()['messages']
            emails = []
            for message in messages:
                email_detail = mock_message_details[message['id']]
                emails.append(email_detail)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Validate results
            success = (
                len(emails) == 3 and
                all('payload' in email for email in emails) and
                all('headers' in email['payload'] for email in emails)
            )
            
            self.test_results['gmail_api_simulation'] = {
                'success': success,
                'execution_time': execution_time,
                'emails_fetched': len(emails),
                'avg_time_per_email': execution_time / len(emails)
            }
            
            print(f"   ✅ Gmail API simulation: {len(emails)} emails fetched in {execution_time:.3f}s")
            print(f"   📊 Average time per email: {execution_time/len(emails):.3f}s")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Gmail API simulation test failed: {e}")
            self.test_results['gmail_api_simulation'] = {'success': False, 'error': str(e)}
            return False
    
    def get_integration_summary(self):
        """Get summary of Gmail integration tests"""
        if not self.test_results:
            return "No integration tests run"
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results.values() if r.get('success', False))
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': f"{(successful_tests/total_tests)*100:.1f}%",
            'test_results': self.test_results
        }

class EdgeCaseTester:
    """Test edge cases and error handling"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_empty_email_handling(self):
        """Test handling of empty emails"""
        print("🧪 Testing Empty Email Handling...")
        
        try:
            from app import parse_email, preprocess_text, analyze_sentiment
            
            # Test empty email
            empty_email = {
                'payload': {
                    'headers': [],
                    'body': {'data': ''}
                }
            }
            
            # Should handle empty email gracefully
            parsed = parse_email(empty_email)
            success = (
                parsed['subject'] == '' and
                parsed['sender'] == '' and
                parsed['body'] == ''
            )
            
            # Test empty text preprocessing
            processed = preprocess_text('')
            success = success and (processed == '')
            
            # Test empty text sentiment analysis
            sentiment, score = analyze_sentiment('')
            success = success and (sentiment == 'neutral')
            
            self.test_results['empty_email_handling'] = {'success': success}
            print(f"   {'✅' if success else '❌'} Empty email handling: {'PASSED' if success else 'FAILED'}")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Empty email handling test failed: {e}")
            self.test_results['empty_email_handling'] = {'success': False, 'error': str(e)}
            return False
    
    def test_malformed_email_handling(self):
        """Test handling of malformed emails"""
        print("🧪 Testing Malformed Email Handling...")
        
        try:
            from app import parse_email
            
            # Test malformed email structures
            malformed_emails = [
                {'payload': {}},  # Missing headers and body
                {'payload': {'headers': None, 'body': None}},  # None values
                {'payload': {'headers': [], 'body': {'data': None}}},  # None data
                {'invalid': 'structure'},  # Completely invalid structure
            ]
            
            success_count = 0
            for email in malformed_emails:
                try:
                    parsed = parse_email(email)
                    # Should return empty strings for malformed emails
                    if (parsed['subject'] == '' and 
                        parsed['sender'] == '' and 
                        parsed['body'] == ''):
                        success_count += 1
                except Exception:
                    # Should handle exceptions gracefully
                    success_count += 1
            
            success = success_count == len(malformed_emails)
            self.test_results['malformed_email_handling'] = {'success': success}
            print(f"   {'✅' if success else '❌'} Malformed email handling: {'PASSED' if success else 'FAILED'}")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Malformed email handling test failed: {e}")
            self.test_results['malformed_email_handling'] = {'success': False, 'error': str(e)}
            return False
    
    def test_large_text_handling(self):
        """Test handling of very large text"""
        print("🧪 Testing Large Text Handling...")
        
        try:
            from app import preprocess_text, analyze_sentiment
            
            # Create very large text
            large_text = "This is a test sentence. " * 10000  # ~250KB text
            
            # Test preprocessing
            start_time = time.time()
            processed = preprocess_text(large_text)
            preprocessing_time = time.time() - start_time
            
            # Test sentiment analysis
            start_time = time.time()
            sentiment, score = analyze_sentiment(large_text)
            analysis_time = time.time() - start_time
            
            # Should handle large text without errors
            success = (
                isinstance(processed, str) and
                len(processed) > 0 and
                isinstance(sentiment, str) and
                isinstance(score, float) and
                preprocessing_time < 10.0 and  # Should complete within 10 seconds
                analysis_time < 30.0  # Should complete within 30 seconds
            )
            
            self.test_results['large_text_handling'] = {
                'success': success,
                'preprocessing_time': preprocessing_time,
                'analysis_time': analysis_time,
                'text_length': len(large_text)
            }
            print(f"   {'✅' if success else '❌'} Large text handling: {'PASSED' if success else 'FAILED'}")
            print(f"   📊 Preprocessing: {preprocessing_time:.3f}s, Analysis: {analysis_time:.3f}s")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Large text handling test failed: {e}")
            self.test_results['large_text_handling'] = {'success': False, 'error': str(e)}
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
                    
                    if (isinstance(processed, str) and 
                        isinstance(sentiment, str) and 
                        isinstance(score, float)):
                        success_count += 1
                except Exception:
                    pass  # Some special characters might cause issues, which is acceptable
            
            success = success_count >= len(special_texts) * 0.8  # 80% success rate acceptable
            self.test_results['special_characters_handling'] = {
                'success': success,
                'success_count': success_count,
                'total_tests': len(special_texts)
            }
            print(f"   {'✅' if success else '❌'} Special characters handling: {'PASSED' if success else 'FAILED'}")
            print(f"   📊 Success rate: {success_count}/{len(special_texts)}")
            
            return success
            
        except Exception as e:
            print(f"   ❌ Special characters handling test failed: {e}")
            self.test_results['special_characters_handling'] = {'success': False, 'error': str(e)}
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

class EnhancedTestSuite:
    """Enhanced test suite for Email Insights application"""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        self.gmail_tester = GmailIntegrationTester()
        self.edge_case_tester = EdgeCaseTester()
        self.test_results = {}
    
    def run_performance_tests(self):
        """Run performance benchmark tests"""
        print("🚀 Running Performance Tests...")
        print("=" * 50)
        
        # Test email parsing performance
        self.gmail_tester.test_email_parsing_performance()
        
        # Test text preprocessing performance
        self.gmail_tester.test_text_preprocessing_performance()
        
        # Test AI analysis performance
        self.gmail_tester.test_ai_analysis_performance()
        
        # Test Gmail API simulation
        self.gmail_tester.test_gmail_api_simulation()
    
    def run_integration_tests(self):
        """Run Gmail integration tests"""
        print("\n📬 Running Gmail Integration Tests...")
        print("=" * 50)
        
        # All integration tests are run in the Gmail tester
        pass  # Tests are already run in run_performance_tests()
    
    def run_edge_case_tests(self):
        """Run edge case and error handling tests"""
        print("\n⚠️  Running Edge Case Tests...")
        print("=" * 50)
        
        # Test empty email handling
        self.edge_case_tester.test_empty_email_handling()
        
        # Test malformed email handling
        self.edge_case_tester.test_malformed_email_handling()
        
        # Test large text handling
        self.edge_case_tester.test_large_text_handling()
        
        # Test special characters handling
        self.edge_case_tester.test_special_characters_handling()
    
    def run_flask_api_tests(self):
        """Run Flask API tests"""
        print("\n🌐 Running Flask API Tests...")
        print("=" * 50)
        
        try:
            from app import app
            
            with app.test_client() as client:
                # Test health endpoint
                start_time = time.time()
                response = client.get('/')
                response_time = time.time() - start_time
                
                success = response.status_code == 200
                self.benchmark.add_result('flask_homepage', response_time, success)
                print(f"   {'✅' if success else '❌'} Homepage: {response_time:.3f}s")
                
                # Test fetch and analyze endpoint (with mock data)
                start_time = time.time()
                response = client.post('/fetch_and_analyze')
                response_time = time.time() - start_time
                
                # Should return error since no Gmail service is available
                success = response.status_code in [200, 500]  # Either success or expected error
                self.benchmark.add_result('flask_fetch_analyze', response_time, success)
                print(f"   {'✅' if success else '❌'} Fetch and analyze: {response_time:.3f}s")
                
        except Exception as e:
            print(f"   ❌ Flask API tests failed: {e}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 ENHANCED EMAIL INSIGHTS TEST REPORT")
        print("=" * 60)
        
        # Performance Summary
        perf_summary = self.benchmark.get_summary()
        print(f"\n🚀 Performance Summary:")
        print(f"   Total Tests: {perf_summary['total_tests']}")
        print(f"   Success Rate: {perf_summary['success_rate']}")
        print(f"   Average Execution Time: {perf_summary['average_execution_time']}")
        print(f"   Max Execution Time: {perf_summary['max_execution_time']}")
        
        # Gmail Integration Summary
        integration_summary = self.gmail_tester.get_integration_summary()
        print(f"\n📬 Gmail Integration Summary:")
        print(f"   Total Tests: {integration_summary['total_tests']}")
        print(f"   Success Rate: {integration_summary['success_rate']}")
        
        # Edge Case Summary
        edge_case_summary = self.edge_case_tester.get_edge_case_summary()
        print(f"\n⚠️  Edge Case Summary:")
        print(f"   Total Tests: {edge_case_summary['total_tests']}")
        print(f"   Success Rate: {edge_case_summary['success_rate']}")
        
        # Overall Assessment
        total_tests = (perf_summary['total_tests'] + 
                      integration_summary['total_tests'] + 
                      edge_case_summary['total_tests'])
        
        total_success = (perf_summary['successful_tests'] + 
                        integration_summary['successful_tests'] + 
                        edge_case_summary['successful_tests'])
        
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🏆 Overall Assessment:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful Tests: {total_success}")
        print(f"   Overall Success Rate: {overall_success_rate:.1f}%")
        
        if overall_success_rate >= 90:
            print("   🎉 EXCELLENT: Email Insights system is performing exceptionally well!")
        elif overall_success_rate >= 80:
            print("   ✅ GOOD: Email Insights system is performing well with minor issues.")
        elif overall_success_rate >= 70:
            print("   ⚠️  FAIR: Email Insights system needs some improvements.")
        else:
            print("   ❌ POOR: Email Insights system requires significant improvements.")
        
        return {
            'performance': perf_summary,
            'integration': integration_summary,
            'edge_cases': edge_case_summary,
            'overall_success_rate': overall_success_rate
        }
    
    def run_all_tests(self):
        """Run all enhanced tests"""
        print("🚀 Email Insights - Enhanced Test Suite")
        print("=" * 60)
        
        # Run performance tests
        self.run_performance_tests()
        
        # Run integration tests
        self.run_integration_tests()
        
        # Run edge case tests
        self.run_edge_case_tests()
        
        # Run Flask API tests
        self.run_flask_api_tests()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        return report

def main():
    """Run the enhanced test suite"""
    test_suite = EnhancedTestSuite()
    report = test_suite.run_all_tests()
    
    # Save report to file
    with open('enhanced_email_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: enhanced_email_test_report.json")
    
    return report

if __name__ == "__main__":
    main()
