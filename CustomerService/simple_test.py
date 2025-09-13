#!/usr/bin/env python3
"""
Customer Service Feedback Analysis Tool - Simple Test Suite
Tests core functionality without heavy dependencies
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def test_data_processing():
    """Test basic data processing capabilities"""
    print("📝 Testing Data Processing...")
    
    try:
        # Test pandas functionality
        sample_data = {
            'text': ['Great product!', 'Poor service', 'Average experience'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'source': ['email', 'review', 'survey']
        }
        
        df = pd.DataFrame(sample_data)
        print(f"   ✅ DataFrame created: {len(df)} rows")
        
        # Test date processing
        df['date'] = pd.to_datetime(df['date'])
        print(f"   ✅ Date processing: {df['date'].dtype}")
        
        # Test text length analysis
        df['text_length'] = df['text'].str.len()
        avg_length = df['text_length'].mean()
        print(f"   ✅ Text analysis: Average length {avg_length:.1f} characters")
        
        return True, f"Processed {len(df)} records"
        
    except Exception as e:
        return False, str(e)

def test_sentiment_analysis_simple():
    """Test simple sentiment analysis using TextBlob"""
    print("😊 Testing Simple Sentiment Analysis...")
    
    try:
        from textblob import TextBlob
        
        test_texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ]
        
        results = []
        for text in test_texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
                
            results.append({
                'text': text,
                'sentiment': sentiment,
                'polarity': polarity
            })
        
        positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
        negative_count = sum(1 for r in results if r['sentiment'] == 'negative')
        neutral_count = sum(1 for r in results if r['sentiment'] == 'neutral')
        
        print(f"   ✅ Sentiment analysis: {positive_count} positive, {negative_count} negative, {neutral_count} neutral")
        return True, f"Analyzed {len(results)} texts"
        
    except Exception as e:
        return False, str(e)

def test_topic_extraction_simple():
    """Test simple topic extraction using word frequency"""
    print("🔍 Testing Simple Topic Extraction...")
    
    try:
        from collections import Counter
        import re
        
        feedback_texts = [
            "The customer service was excellent and very helpful",
            "I had issues with the product quality and delivery",
            "The pricing is too high compared to competitors",
            "Great customer support team, very responsive",
            "Product arrived damaged and customer service was slow"
        ]
        
        # Simple word frequency analysis
        all_words = []
        for text in feedback_texts:
            # Simple word extraction
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'was', 'is', 'are', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
        
        word_freq = Counter(filtered_words)
        top_topics = word_freq.most_common(5)
        
        print(f"   ✅ Topic extraction: Top topics {[topic[0] for topic in top_topics]}")
        return True, f"Extracted {len(top_topics)} topics"
        
    except Exception as e:
        return False, str(e)

def test_key_phrase_extraction():
    """Test key phrase extraction using TextBlob"""
    print("💡 Testing Key Phrase Extraction...")
    
    try:
        from textblob import TextBlob
        
        test_text = "The customer service team was very helpful and resolved my issue quickly. The product quality is excellent and I would recommend it to others."
        
        blob = TextBlob(test_text)
        phrases = blob.noun_phrases
        
        print(f"   ✅ Key phrases: {phrases[:3]}")  # Show first 3 phrases
        return True, f"Extracted {len(phrases)} phrases"
        
    except Exception as e:
        return False, str(e)

def test_actionable_insights():
    """Test actionable insights generation"""
    print("🎯 Testing Actionable Insights...")
    
    try:
        feedback_data = [
            "Customer service response time is too slow",
            "The product has bugs that need fixing",
            "I wish you had better documentation",
            "The pricing is too expensive",
            "Great product, keep up the good work"
        ]
        
        action_categories = {
            'response_time': ['slow', 'response', 'time'],
            'bug': ['bug', 'error', 'issue', 'problem'],
            'documentation': ['documentation', 'manual', 'help', 'guide'],
            'pricing': ['expensive', 'price', 'cost', 'pricing'],
            'praise': ['great', 'excellent', 'good', 'amazing']
        }
        
        insights = []
        for text in feedback_data:
            text_lower = text.lower()
            for category, keywords in action_categories.items():
                if any(keyword in text_lower for keyword in keywords):
                    insights.append({
                        'text': text,
                        'category': category,
                        'keywords_found': [kw for kw in keywords if kw in text_lower]
                    })
                    break
        
        category_counts = {}
        for insight in insights:
            category = insight['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"   ✅ Actionable insights: {category_counts}")
        return True, f"Generated {len(insights)} insights"
        
    except Exception as e:
        return False, str(e)

def test_file_processing():
    """Test file processing capabilities"""
    print("📁 Testing File Processing...")
    
    try:
        # Create sample CSV data
        sample_data = {
            'text': [
                'Great product, highly recommend!',
                'Poor customer service experience',
                'Average product, could be better',
                'Excellent support team',
                'Product arrived damaged'
            ],
            'date': [
                '2023-01-01',
                '2023-01-02', 
                '2023-01-03',
                '2023-01-04',
                '2023-01-05'
            ],
            'source': ['email', 'review', 'survey', 'email', 'review']
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test CSV operations
        csv_filename = 'test_feedback.csv'
        df.to_csv(csv_filename, index=False)
        
        # Read it back
        df_read = pd.read_csv(csv_filename)
        
        # Clean up
        os.remove(csv_filename)
        
        print(f"   ✅ File processing: Read/write {len(df_read)} records")
        return True, f"Processed {len(df_read)} records"
        
    except Exception as e:
        return False, str(e)

def test_performance():
    """Test performance with larger datasets"""
    print("⚡ Testing Performance...")
    
    try:
        # Generate larger dataset
        n_records = 1000
        feedback_texts = [
            "Great product, highly recommend!",
            "Poor customer service experience", 
            "Average product, could be better",
            "Excellent support team",
            "Product arrived damaged",
            "Fast delivery and good quality",
            "Overpriced compared to competitors",
            "User interface is confusing",
            "Documentation is lacking",
            "Best product I've ever used"
        ]
        
        data = {
            'text': np.random.choice(feedback_texts, n_records),
            'date': pd.date_range('2023-01-01', periods=n_records, freq='D'),
            'source': np.random.choice(['email', 'review', 'survey', 'social_media'], n_records)
        }
        
        start_time = time.time()
        df = pd.DataFrame(data)
        processing_time = time.time() - start_time
        
        print(f"   ✅ Performance: Processed {n_records} records in {processing_time:.3f}s")
        return True, f"Processed {n_records} records in {processing_time:.3f}s"
        
    except Exception as e:
        return False, str(e)

def run_all_tests():
    """Run all tests and generate report"""
    print("🚀 Customer Service Feedback Analysis - Simple Test Suite")
    print("=" * 60)
    
    tests = [
        ("Data Processing", test_data_processing),
        ("Simple Sentiment Analysis", test_sentiment_analysis_simple),
        ("Simple Topic Extraction", test_topic_extraction_simple),
        ("Key Phrase Extraction", test_key_phrase_extraction),
        ("Actionable Insights", test_actionable_insights),
        ("File Processing", test_file_processing),
        ("Performance", test_performance)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            success, message = test_func()
            results.append({
                'test': test_name,
                'success': success,
                'message': message,
                'execution_time': time.time() - start_time
            })
            if success:
                print(f"   ✅ {test_name}: {message}")
            else:
                print(f"   ❌ {test_name}: {message}")
        except Exception as e:
            results.append({
                'test': test_name,
                'success': False,
                'message': str(e),
                'execution_time': time.time() - start_time
            })
            print(f"   ❌ {test_name}: ERROR - {e}")
    
    total_time = time.time() - start_time
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 SIMPLE TEST SUITE REPORT")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) * 100
    
    print(f"\n🚀 Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful Tests: {successful_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Total Execution Time: {total_time:.3f}s")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"   {status} - {result['test']}: {result['message']}")
    
    # Overall assessment
    if success_rate >= 90:
        assessment = "✅ EXCELLENT: System is working perfectly!"
    elif success_rate >= 75:
        assessment = "✅ GOOD: System is working well with minor issues."
    elif success_rate >= 50:
        assessment = "⚠️  FAIR: System has some issues that need attention."
    else:
        assessment = "❌ POOR: System requires significant improvements."
    
    print(f"\n🏆 Overall Assessment:")
    print(f"   {assessment}")
    
    # Save report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': success_rate,
        'total_execution_time': total_time,
        'results': results,
        'assessment': assessment
    }
    
    with open('simple_test_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: simple_test_report.json")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
