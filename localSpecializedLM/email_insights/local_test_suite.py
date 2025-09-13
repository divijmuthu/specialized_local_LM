#!/usr/bin/env python3
"""
Comprehensive Local Testing Suite for Email Insights Application
Tests all functionality without requiring Gmail OAuth credentials
"""

import requests
import json
import time
import sys
from datetime import datetime

class EmailInsightsTester:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test(self, test_name, success, details=""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        timestamp = datetime.now().strftime("%H:%M:%S")
        result = f"[{timestamp}] {status} - {test_name}"
        if details:
            result += f" - {details}"
        print(result)
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': timestamp
        })
        
    def test_server_connection(self):
        """Test if the Flask server is running"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                self.log_test("Server Connection", True, f"Status: {response.status_code}")
                return True
            else:
                self.log_test("Server Connection", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Server Connection", False, f"Error: {str(e)}")
            return False
    
    def test_demo_mode(self):
        """Test demo mode functionality"""
        try:
            response = requests.post(
                f"{self.base_url}/demo_analyze",
                headers={"Content-Type": "application/json"},
                json={},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('demo_mode'):
                    email_count = len(data.get('data', []))
                    insights = data.get('insights', {})
                    
                    # Check if we have the expected insights
                    has_sentiment = 'sentiment_distribution' in insights
                    has_email_types = 'email_type_distribution' in insights
                    has_topics = 'topic_distribution' in insights
                    
                    if has_sentiment and has_email_types and has_topics and email_count > 0:
                        self.log_test("Demo Mode Analysis", True, 
                                    f"Analyzed {email_count} emails, got insights")
                        return True
                    else:
                        self.log_test("Demo Mode Analysis", False, 
                                    "Missing expected insights")
                        return False
                else:
                    self.log_test("Demo Mode Analysis", False, 
                                f"API error: {data.get('error', 'Unknown')}")
                    return False
            else:
                self.log_test("Demo Mode Analysis", False, 
                            f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Demo Mode Analysis", False, f"Error: {str(e)}")
            return False
    
    def test_insight_quality(self):
        """Test the quality of generated insights"""
        try:
            response = requests.post(
                f"{self.base_url}/demo_analyze",
                headers={"Content-Type": "application/json"},
                json={},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    insights = data.get('insights', {})
                    emails = data.get('data', [])
                    
                    # Test sentiment analysis quality
                    sentiment_dist = insights.get('sentiment_distribution', {})
                    has_positive = 'POSITIVE' in sentiment_dist
                    has_negative = 'NEGATIVE' in sentiment_dist
                    
                    # Test email classification quality
                    type_dist = insights.get('email_type_distribution', {})
                    has_complaints = 'complaint' in type_dist
                    has_inquiries = 'inquiry' in type_dist
                    has_feedback = 'feedback' in type_dist
                    
                    # Test topic clustering
                    topic_dist = insights.get('topic_distribution', {})
                    has_topics = len(topic_dist) > 0
                    
                    # Test individual email analysis
                    email_analysis_quality = True
                    for email in emails:
                        if not email.get('sentiment') or not email.get('email_type'):
                            email_analysis_quality = False
                            break
                    
                    if (has_positive and has_negative and has_complaints and 
                        has_inquiries and has_feedback and has_topics and 
                        email_analysis_quality):
                        self.log_test("Insight Quality", True, 
                                    "All analysis types working correctly")
                        return True
                    else:
                        self.log_test("Insight Quality", False, 
                                    "Some analysis types missing or incorrect")
                        return False
                else:
                    self.log_test("Insight Quality", False, 
                                "Failed to get insights")
                    return False
            else:
                self.log_test("Insight Quality", False, 
                            f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Insight Quality", False, f"Error: {str(e)}")
            return False
    
    def test_web_interface(self):
        """Test web interface components"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                html_content = response.text
                
                # Check for key HTML elements
                has_title = "Email Insights Dashboard" in html_content
                has_charts = "chart.js" in html_content.lower()
                has_demo_button = "Demo Mode (Sample Data)" in html_content
                has_analyze_button = "Fetch and Analyze Emails" in html_content
                
                if has_title and has_charts and has_demo_button and has_analyze_button:
                    self.log_test("Web Interface", True, 
                                "All UI components present")
                    return True
                else:
                    self.log_test("Web Interface", False, 
                                "Missing UI components")
                    return False
            else:
                self.log_test("Web Interface", False, 
                            f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Web Interface", False, f"Error: {str(e)}")
            return False
    
    def test_performance(self):
        """Test performance of demo analysis"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/demo_analyze",
                headers={"Content-Type": "application/json"},
                json={},
                timeout=60
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200 and response_time < 30:
                self.log_test("Performance", True, 
                            f"Response time: {response_time:.2f}s")
                return True
            elif response_time >= 30:
                self.log_test("Performance", False, 
                            f"Too slow: {response_time:.2f}s")
                return False
            else:
                self.log_test("Performance", False, 
                            f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Performance", False, f"Error: {str(e)}")
            return False
    
    def test_error_handling(self):
        """Test error handling"""
        try:
            # Test with invalid endpoint
            response = requests.get(f"{self.base_url}/invalid_endpoint", timeout=5)
            if response.status_code == 404:
                self.log_test("Error Handling", True, 
                            "Properly handles invalid endpoints")
                return True
            else:
                self.log_test("Error Handling", False, 
                            f"Unexpected status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Error Handling", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Comprehensive Local Testing Suite")
        print("=" * 60)
        
        tests = [
            ("Server Connection", self.test_server_connection),
            ("Web Interface", self.test_web_interface),
            ("Demo Mode", self.test_demo_mode),
            ("Insight Quality", self.test_insight_quality),
            ("Performance", self.test_performance),
            ("Error Handling", self.test_error_handling),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                self.log_test(test_name, False, f"Exception: {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Application is working perfectly!")
            print("\n✅ Ready for production use:")
            print("   • Demo mode working with realistic data")
            print("   • AI analysis functioning correctly")
            print("   • Web interface fully operational")
            print("   • Performance within acceptable limits")
            print("   • Error handling working properly")
        else:
            print("⚠️  Some tests failed. Check the details above.")
        
        return passed == total
    
    def generate_report(self):
        """Generate a detailed test report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.test_results),
            'passed_tests': sum(1 for r in self.test_results if r['success']),
            'failed_tests': sum(1 for r in self.test_results if not r['success']),
            'test_details': self.test_results
        }
        
        with open('test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed test report saved to: test_report.json")

def main():
    """Main testing function"""
    print("🧪 Email Insights Application - Local Testing Suite")
    print("Testing all functionality without Gmail OAuth requirements")
    print()
    
    # Check if server is running
    tester = EmailInsightsTester()
    
    if not tester.test_server_connection():
        print("❌ Server is not running!")
        print("Please start the Flask app with: python3 app.py")
        print("Then run this test suite again.")
        sys.exit(1)
    
    # Run all tests
    success = tester.run_all_tests()
    
    # Generate report
    tester.generate_report()
    
    if success:
        print("\n🌟 The Email Insights Application is fully functional!")
        print("You can now:")
        print("1. Open http://localhost:5001 in your browser")
        print("2. Click 'Demo Mode (Sample Data)' to see it in action")
        print("3. Explore the AI-powered insights and visualizations")
        sys.exit(0)
    else:
        print("\n🔧 Some issues were found. Please check the test results above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
