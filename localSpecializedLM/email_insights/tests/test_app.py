import pytest
import json
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, preprocess_text, analyze_sentiment, classify_email, extract_topics, generate_insights, parse_email

@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_email_data():
    """Sample email data for testing"""
    return {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Test Email Subject'},
                {'name': 'From', 'value': 'test@example.com'},
                {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'}
            ],
            'body': {
                'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded "Test email body content"
            }
        }
    }

@pytest.fixture
def sample_emails():
    """Sample list of emails for testing"""
    return [
        {
            'payload': {
                'headers': [
                    {'name': 'Subject', 'value': 'Complaint about service'},
                    {'name': 'From', 'value': 'customer@example.com'},
                    {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'}
                ],
                'body': {
                    'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
                }
            }
        },
        {
            'payload': {
                'headers': [
                    {'name': 'Subject', 'value': 'Question about product'},
                    {'name': 'From', 'value': 'inquiry@example.com'},
                    {'name': 'Date', 'value': 'Mon, 1 Jan 2024 13:00:00 +0000'}
                ],
                'body': {
                    'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='  # Base64 encoded
                }
            }
        }
    ]

class TestTextProcessing:
    """Test text processing functions"""
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        text = "Hello! This is a TEST email with punctuation, numbers 123, and UPPERCASE."
        result = preprocess_text(text)
        
        # Should be lowercase, no punctuation, no numbers
        assert "hello" in result
        assert "test" in result
        assert "email" in result
        assert "!" not in result
        assert "," not in result
        assert "123" not in result
    
    def test_preprocess_empty_text(self):
        """Test preprocessing empty text"""
        result = preprocess_text("")
        assert result == ""
    
    def test_preprocess_text_with_stopwords(self):
        """Test that stopwords are removed"""
        text = "This is a test email with common words"
        result = preprocess_text(text)
        
        # Split into words for proper testing
        words = result.split()
        
        # Common stopwords should be removed
        assert "this" not in words
        assert "is" not in words
        assert "a" not in words
        assert "with" not in words
        assert "test" in words
        assert "email" in words
        assert "common" in words
        assert "word" in words  # "words" becomes "word" after lemmatization

class TestEmailParsing:
    """Test email parsing functions"""
    
    def test_parse_email(self, sample_email_data):
        """Test email parsing"""
        result = parse_email(sample_email_data)
        
        assert result['subject'] == 'Test Email Subject'
        assert result['sender'] == 'test@example.com'
        assert result['date'] == 'Mon, 1 Jan 2024 12:00:00 +0000'
        assert result['body'] == 'Test email body content'
    
    def test_parse_email_missing_headers(self):
        """Test parsing email with missing headers"""
        email_data = {
            'payload': {
                'headers': [],
                'body': {'data': 'VGVzdCBlbWFpbCBib2R5IGNvbnRlbnQ='}
            }
        }
        result = parse_email(email_data)
        
        assert result['subject'] == ''
        assert result['sender'] == ''
        assert result['date'] == ''
        assert result['body'] == 'Test email body content'

class TestMLAnalysis:
    """Test ML analysis functions"""
    
    @patch('app.sentiment_analyzer')
    def test_analyze_sentiment(self, mock_sentiment_analyzer):
        """Test sentiment analysis"""
        mock_sentiment_analyzer.return_value = [{'label': 'POSITIVE', 'score': 0.95}]
        
        result_label, result_score = analyze_sentiment("This is a great email!")
        
        assert result_label == 'POSITIVE'
        assert result_score == 0.95
        mock_sentiment_analyzer.assert_called_once_with("This is a great email!")
    
    def test_analyze_sentiment_empty_text(self):
        """Test sentiment analysis with empty text"""
        result_label, result_score = analyze_sentiment("")
        
        assert result_label == 'neutral'
        assert result_score == 0.0
    
    @patch('app.classifier')
    def test_classify_email(self, mock_classifier):
        """Test email classification"""
        mock_classifier.return_value = {
            'labels': ['complaint', 'inquiry', 'feedback'],
            'scores': [0.8, 0.15, 0.05]
        }
        
        result_label, result_score = classify_email("I have a complaint about your service")
        
        assert result_label == 'complaint'
        assert result_score == 0.8
        mock_classifier.assert_called_once()
    
    def test_classify_email_empty_text(self):
        """Test email classification with empty text"""
        result_label, result_score = classify_email("")
        
        assert result_label == 'unknown'
        assert result_score == 0.0
    
    def test_extract_topics(self):
        """Test topic extraction"""
        texts = [
            "This is about technology and computers",
            "I love technology and programming",
            "Food and cooking are my hobbies",
            "I enjoy cooking delicious meals"
        ]
        
        clusters = extract_topics(texts, n_clusters=2)
        
        assert len(clusters) == 4
        assert all(isinstance(cluster, (int, np.integer)) for cluster in clusters)
    
    def test_extract_topics_empty_list(self):
        """Test topic extraction with empty list"""
        clusters = extract_topics([])
        assert clusters == []
    
    def test_extract_topics_single_text(self):
        """Test topic extraction with single text"""
        clusters = extract_topics(["Single text"])
        # With only one document, clustering should return empty list
        assert clusters == []

class TestInsightGeneration:
    """Test insight generation"""
    
    def test_generate_insights_empty_dataframe(self):
        """Test insight generation with empty dataframe"""
        df = pd.DataFrame()
        insights, enriched_df = generate_insights(df)
        
        assert insights == {}
        assert enriched_df.empty
    
    @patch('app.analyze_sentiment')
    @patch('app.classify_email')
    @patch('app.extract_topics')
    def test_generate_insights_with_data(self, mock_extract_topics, mock_classify_email, mock_analyze_sentiment):
        """Test insight generation with data"""
        # Setup mocks
        mock_analyze_sentiment.return_value = ('POSITIVE', 0.8)
        mock_classify_email.return_value = ('complaint', 0.7)
        mock_extract_topics.return_value = [0, 1, 0]
        
        # Create test data
        df = pd.DataFrame({
            'cleaned_body': ['test email 1', 'test email 2', 'test email 3']
        })
        
        insights, enriched_df = generate_insights(df)
        
        # Check that insights were generated
        assert 'sentiment_distribution' in insights
        assert 'email_type_distribution' in insights
        assert 'topic_distribution' in insights
        assert 'top_complaints' in insights
        assert 'top_inquiries' in insights
        assert 'top_feedback' in insights
        
        # Check that dataframe was enriched
        assert 'sentiment' in enriched_df.columns
        assert 'email_type' in enriched_df.columns
        assert 'topic_cluster' in enriched_df.columns

class TestFlaskRoutes:
    """Test Flask routes"""
    
    def test_index_route(self, client):
        """Test the index route"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Email Insights Dashboard' in response.data
    
    @patch('app.get_gmail_service')
    @patch('app.fetch_emails')
    @patch('app.preprocess_emails')
    @patch('app.generate_insights')
    def test_fetch_and_analyze_success(self, mock_generate_insights, mock_preprocess_emails, 
                                     mock_fetch_emails, mock_get_gmail_service, client, sample_emails):
        """Test successful fetch and analyze"""
        # Setup mocks
        mock_service = Mock()
        mock_get_gmail_service.return_value = mock_service
        mock_fetch_emails.return_value = sample_emails
        
        df = pd.DataFrame({
            'subject': ['Test 1', 'Test 2'],
            'sender': ['test1@example.com', 'test2@example.com'],
            'date': ['2024-01-01', '2024-01-02'],
            'body': ['Test body 1', 'Test body 2'],
            'cleaned_body': ['test body 1', 'test body 2']
        })
        mock_preprocess_emails.return_value = df
        
        insights = {
            'sentiment_distribution': {'POSITIVE': 2},
            'email_type_distribution': {'complaint': 1, 'inquiry': 1},
            'topic_distribution': {0: 1, 1: 1},
            'top_complaints': ['test body 1'],
            'top_inquiries': ['test body 2'],
            'top_feedback': []
        }
        mock_generate_insights.return_value = (insights, df)
        
        response = client.post('/fetch_and_analyze')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is True
        assert 'insights' in data
        assert 'data' in data
    
    @patch('app.get_gmail_service')
    @patch('app.fetch_emails')
    def test_fetch_and_analyze_no_emails(self, mock_fetch_emails, mock_get_gmail_service, client):
        """Test fetch and analyze with no emails"""
        mock_service = Mock()
        mock_get_gmail_service.return_value = mock_service
        mock_fetch_emails.return_value = []
        
        response = client.post('/fetch_and_analyze')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is False
        assert 'No emails found' in data['error']
    
    @patch('app.get_gmail_service')
    def test_fetch_and_analyze_gmail_error(self, mock_get_gmail_service, client):
        """Test fetch and analyze with Gmail service error"""
        mock_get_gmail_service.side_effect = Exception("Gmail API error")
        
        response = client.post('/fetch_and_analyze')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['success'] is False
        assert 'Gmail API error' in data['error']
    
    def test_train_model_endpoint_exists(self, client):
        """Test that train model endpoint exists and returns proper structure"""
        response = client.post('/train_model')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert 'success' in data
        assert isinstance(data['success'], bool)
        # Should fail without credentials, but structure should be correct
        if not data['success']:
            assert 'error' in data

class TestIntegration:
    """Integration tests"""
    
    def test_app_initialization(self):
        """Test that the app initializes correctly"""
        assert app is not None
        # In test environment, TESTING might be True
        assert 'TESTING' in app.config
    
    def test_imports(self):
        """Test that all required modules can be imported"""
        try:
            from app import (
                Flask, render_template, request, jsonify,
                get_gmail_service, fetch_emails, parse_email,
                preprocess_text, analyze_sentiment, classify_email,
                extract_topics, generate_insights
            )
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")

if __name__ == '__main__':
    pytest.main([__file__])
