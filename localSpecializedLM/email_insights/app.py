from flask import Flask, render_template, request, jsonify
import os
import pickle
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import base64
import re
from sample_emails import get_sample_emails, get_sample_email_bodies

# Initialize Flask app
app = Flask(__name__)

# Define the scopes for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Load pre-trained models
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    print(f"Warning: Could not load pre-trained models: {e}")
    tokenizer = None
    model = None
    sentiment_analyzer = None
    classifier = None

def get_gmail_service():
    """Get Gmail service with OAuth authentication"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    service = build('gmail', 'v1', credentials=creds)
    return service

def fetch_emails(service, user_id='me', label_ids=['INBOX'], max_results=10):
    """Fetch emails from Gmail"""
    try:
        results = service.users().messages().list(
            userId=user_id, 
            labelIds=label_ids,
            maxResults=max_results
        ).execute()
        messages = results.get('messages', [])
        emails = []
        
        if not messages:
            print('No messages found.')
        else:
            for message in messages:
                msg = service.users().messages().get(userId=user_id, id=message['id']).execute()
                emails.append(msg)
        
        return emails
    except Exception as error:
        print(f'An error occurred: {error}')
        return []

def parse_email(email, email_index=None):
    """Parse email content from Gmail API response"""
    headers = email['payload']['headers']
    subject = next((header['value'] for header in headers if header['name'] == 'Subject'), '')
    sender = next((header['value'] for header in headers if header['name'] == 'From'), '')
    date = next((header['value'] for header in headers if header['name'] == 'Date'), '')
    
    body = ''
    if 'parts' in email['payload']:
        for part in email['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                body_data = part['body']['data']
                body = body_data
                break
    else:
        body_data = email['payload']['body']['data']
        body = body_data
    
    try:
        decoded_body = base64.urlsafe_b64decode(body).decode('utf-8')
    except:
        decoded_body = ''
    
    # If this is a demo email, use realistic content
    if email_index is not None and email_index < len(get_sample_email_bodies()):
        decoded_body = get_sample_email_bodies()[email_index]
    
    soup = BeautifulSoup(decoded_body, 'html.parser')
    clean_body = soup.get_text()
    clean_body = re.sub(r'\s+', ' ', clean_body).strip()
    
    return {
        'subject': subject,
        'sender': sender,
        'date': date,
        'body': clean_body
    }

def preprocess_text(text):
    """Preprocess text for analysis"""
    text = text.lower()
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    # Remove stopwords and empty tokens
    tokens = [word for word in tokens if word not in stop_words and word.strip()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def preprocess_emails(emails, is_demo=False):
    """Preprocess all emails"""
    if is_demo:
        parsed_emails = [parse_email(email, i) for i, email in enumerate(emails)]
    else:
        parsed_emails = [parse_email(email) for email in emails]
    df = pd.DataFrame(parsed_emails)
    df['cleaned_body'] = df['body'].apply(preprocess_text)
    return df

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    if not text.strip():
        return 'neutral', 0.0
    
    if sentiment_analyzer is None:
        return 'neutral', 0.0
    
    try:
        result = sentiment_analyzer(text)[0]
        return result['label'], result['score']
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 'neutral', 0.0

def classify_email(text, candidate_labels=["complaint", "inquiry", "feedback"]):
    """Classify email type"""
    if not text.strip():
        return 'unknown', 0.0
    
    if classifier is None:
        return 'unknown', 0.0
    
    try:
        result = classifier(text, candidate_labels)
        return result['labels'][0], result['scores'][0]
    except Exception as e:
        print(f"Error in email classification: {e}")
        return 'unknown', 0.0

def extract_topics(texts, n_clusters=5):
    """Extract topics using TF-IDF and K-means"""
    if not texts or len(texts) == 0:
        return []
    
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    try:
        tfidf = vectorizer.fit_transform(texts)
        if tfidf.shape[0] > 0:
            kmeans = KMeans(n_clusters=min(n_clusters, tfidf.shape[0]), random_state=42)
            clusters = kmeans.fit_predict(tfidf)
            return clusters
        else:
            return []
    except Exception as e:
        print(f"Error in topic extraction: {e}")
        return []

def generate_insights(df):
    """Generate insights from email data"""
    if df.empty:
        return {}, df
    
    try:
        df[['sentiment', 'sentiment_score']] = df['cleaned_body'].apply(
            lambda x: pd.Series(analyze_sentiment(x))
        )
        df[['email_type', 'type_score']] = df['cleaned_body'].apply(
            lambda x: pd.Series(classify_email(x))
        )
        df['topic_cluster'] = extract_topics(df['cleaned_body'].tolist())
    except Exception as e:
        print(f"Error in generating insights: {e}")
    
    insights = {
        'sentiment_distribution': dict(df['sentiment'].value_counts()) if not df.empty else {},
        'email_type_distribution': dict(df['email_type'].value_counts()) if not df.empty else {},
        'topic_distribution': dict(df['topic_cluster'].value_counts()) if not df.empty else {},
        'top_complaints': df[df['email_type'] == 'complaint']['cleaned_body'].tolist() if not df.empty else [],
        'top_inquiries': df[df['email_type'] == 'inquiry']['cleaned_body'].tolist() if not df.empty else [],
        'top_feedback': df[df['email_type'] == 'feedback']['cleaned_body'].tolist() if not df.empty else [],
    }
    return insights, df

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/fetch_and_analyze', methods=['POST'])
def fetch_and_analyze():
    """Fetch and analyze emails endpoint"""
    try:
        service = get_gmail_service()
        emails = fetch_emails(service)
        
        if not emails:
            return jsonify({
                'success': False,
                'error': 'No emails found or error fetching emails.'
            })
        
        df = preprocess_emails(emails)
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'No valid emails to process.'
            })
        
        insights, enriched_df = generate_insights(df)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        insights = convert_numpy_types(insights)
        data_records = enriched_df.to_dict('records')
        data_records = convert_numpy_types(data_records)
        
        return jsonify({
            'success': True,
            'insights': insights,
            'data': data_records
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/demo_analyze', methods=['POST'])
def demo_analyze():
    """Demo mode - analyze sample emails"""
    try:
        # Get sample emails
        emails = get_sample_emails()
        
        if not emails:
            return jsonify({
                'success': False,
                'error': 'No sample emails available.'
            })
        
        df = preprocess_emails(emails, is_demo=True)
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'No valid emails to process.'
            })
        
        insights, enriched_df = generate_insights(df)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        insights = convert_numpy_types(insights)
        data_records = enriched_df.to_dict('records')
        data_records = convert_numpy_types(data_records)
        
        return jsonify({
            'success': True,
            'insights': insights,
            'data': data_records,
            'demo_mode': True,
            'message': f'Analyzed {len(emails)} sample emails in demo mode'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def train_model_route():
    """Train the model on email data"""
    try:
        service = get_gmail_service()
        emails = fetch_emails(service)
        
        if not emails:
            return jsonify({
                'success': False,
                'error': 'No emails found or error fetching emails.'
            })
        
        df = preprocess_emails(emails)
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'No valid emails to process.'
            })
        
        # For training, we need labeled data. Here, we'll create dummy labels for demonstration.
        # In a real scenario, you would have labeled emails.
        df['label'] = ['complaint', 'inquiry', 'feedback'] * (len(df) // 3 + 1)
        df['label'] = df['label'][:len(df)]
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['label_encoded'] = label_encoder.fit_transform(df['label'])
        
        # Tokenize data
        texts = df['cleaned_body'].tolist()
        labels = df['label_encoded'].tolist()
        
        if tokenizer is None:
            return jsonify({
                'success': False,
                'error': 'Tokenizer not available. Please check model loading.'
            })
        
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        class EmailDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.labels)
        
        dataset = EmailDataset(inputs, labels)
        
        if len(dataset) == 0:
            return jsonify({
                'success': False,
                'error': 'No valid data for training.'
            })
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )
        
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not available. Please check model loading.'
            })
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        trainer.train()
        
        # Save the trained model and tokenizer
        model.save_pretrained('./saved_model')
        tokenizer.save_pretrained('./saved_model')
        
        return jsonify({
            'success': True,
            'message': 'Model trained and saved successfully.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    """Train model endpoint"""
    return train_model_route()

if __name__ == '__main__':
    app.run(debug=True, port=5001)
