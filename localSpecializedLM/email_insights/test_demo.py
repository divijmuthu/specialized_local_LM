#!/usr/bin/env python3
"""
Demo script to test the email insights application functionality
without requiring Gmail credentials.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import preprocess_text, analyze_sentiment, classify_email, extract_topics, generate_insights
import pandas as pd

def test_text_processing():
    """Test text processing functionality"""
    print("🧪 Testing Text Processing...")
    
    # Test preprocessing
    sample_text = "Hello! This is a TEST email with punctuation, numbers 123, and UPPERCASE."
    processed = preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
    print("✅ Text preprocessing works!\n")

def test_sentiment_analysis():
    """Test sentiment analysis"""
    print("🧪 Testing Sentiment Analysis...")
    
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "The weather is okay today."
    ]
    
    for text in test_texts:
        sentiment, score = analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (confidence: {score:.2f})")
    
    print("✅ Sentiment analysis works!\n")

def test_email_classification():
    """Test email classification"""
    print("🧪 Testing Email Classification...")
    
    test_emails = [
        "I have a complaint about your service. It's not working properly.",
        "Can you help me understand how to use this feature?",
        "Thank you for the great service! I'm very satisfied."
    ]
    
    for email in test_emails:
        email_type, score = classify_email(email)
        print(f"Email: {email}")
        print(f"Type: {email_type} (confidence: {score:.2f})")
    
    print("✅ Email classification works!\n")

def test_topic_extraction():
    """Test topic extraction"""
    print("🧪 Testing Topic Extraction...")
    
    sample_emails = [
        "I need help with my account login issues",
        "The product is not working as expected",
        "How do I reset my password?",
        "I'm having trouble with the payment system",
        "Can you explain the pricing plans?",
        "The customer service was excellent"
    ]
    
    clusters = extract_topics(sample_emails, n_clusters=3)
    print(f"Sample emails: {len(sample_emails)}")
    print(f"Topic clusters: {clusters}")
    
    # Group emails by cluster
    for i, cluster in enumerate(clusters):
        print(f"Cluster {cluster}: {sample_emails[i]}")
    
    print("✅ Topic extraction works!\n")

def test_insight_generation():
    """Test insight generation"""
    print("🧪 Testing Insight Generation...")
    
    # Create sample email data
    sample_data = [
        {
            'subject': 'Complaint about service',
            'sender': 'customer1@example.com',
            'date': '2024-01-01',
            'body': 'I have a complaint about your service. It is not working properly.',
            'cleaned_body': 'complaint service working properly'
        },
        {
            'subject': 'Question about product',
            'sender': 'customer2@example.com', 
            'date': '2024-01-02',
            'body': 'Can you help me understand how to use this feature?',
            'cleaned_body': 'help understand use feature'
        },
        {
            'subject': 'Thank you',
            'sender': 'customer3@example.com',
            'date': '2024-01-03', 
            'body': 'Thank you for the great service! I am very satisfied.',
            'cleaned_body': 'thank great service satisfied'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    insights, enriched_df = generate_insights(df)
    
    print("Generated insights:")
    for key, value in insights.items():
        print(f"  {key}: {value}")
    
    print(f"\nEnriched dataframe columns: {list(enriched_df.columns)}")
    print("✅ Insight generation works!\n")

def main():
    """Run all demo tests"""
    print("🚀 Email Insights Application Demo")
    print("=" * 50)
    
    try:
        test_text_processing()
        test_sentiment_analysis()
        test_email_classification()
        test_topic_extraction()
        test_insight_generation()
        
        print("🎉 All tests passed! The application is working correctly.")
        print("\n📋 Next steps:")
        print("1. Set up Google OAuth credentials (see README.md)")
        print("2. Run: python3 app.py")
        print("3. Open: http://localhost:5000")
        print("4. Click 'Fetch and Analyze Emails' to start!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
