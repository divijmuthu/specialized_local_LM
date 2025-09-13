# Customer Feedback Analysis Tool
# Implementation based on specter.txt spec

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging
from logging.handlers import RotatingFileHandler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from textblob import TextBlob
from collections import Counter
import datetime

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('feedback_analysis.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

sample_feedback_data = [
    {"source": "email", "date": "2023-01-01", "text": "I love your product! It's amazing.", "sentiment": "positive", "topic": "product_quality"},
    {"source": "review", "date": "2023-01-02", "text": "Customer service was terrible. They didn't respond to my issue.", "sentiment": "negative", "topic": "customer_service"},
    {"source": "survey", "date": "2023-01-03", "text": "The new feature is great! It solves a major problem for us.", "sentiment": "positive", "topic": "product_feature"},
    {"source": "social_media", "date": "2023-01-04", "text": "I've been a customer for years, but the recent changes have been disappointing.", "sentiment": "negative", "topic": "product_change"},
    {"source": "email", "date": "2023-02-01", "text": "The support team was very helpful in resolving my issue quickly.", "sentiment": "positive", "topic": "customer_service"},
    {"source": "review", "date": "2023-02-05", "text": "The product is good, but the pricing is too high compared to competitors.", "sentiment": "neutral", "topic": "pricing"},
    {"source": "survey", "date": "2023-02-10", "text": "I would recommend this product to others. It's reliable and effective.", "sentiment": "positive", "topic": "product_quality"},
    {"source": "social_media", "date": "2023-02-15", "text": "Just had the worst experience with your customer service. Never buying from you again!", "sentiment": "negative", "topic": "customer_service"},
    {"source": "email", "date": "2023-03-01", "text": "The new update is fantastic! Much better than before.", "sentiment": "positive", "topic": "product_update"},
    {"source": "review", "date": "2023-03-05", "text": "The product works well, but the documentation is lacking.", "sentiment": "neutral", "topic": "documentation"},
    {"source": "survey", "date": "2023-03-10", "text": "I've been very satisfied with your product and service.", "sentiment": "positive", "topic": "overall_satisfaction"},
    {"source": "social_media", "date": "2023-03-15", "text": "Your product is overpriced and not worth the cost.", "sentiment": "negative", "topic": "pricing"},
]

# ...existing code...
# --- Spec Functions ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def analyze_sentiment(text):
    if not text.strip():
        return 'neutral', 0.0
    try:
        result = sentiment_analyzer(text)[0]
        return result['label'], result['score']
    except Exception as e:
        app.logger.error(f"Error in sentiment analysis: {e}")
        return 'neutral', 0.0

def extract_topics(texts, n_topics=5):
    if not texts or len(texts) == 0:
        return [], []
    try:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = vectorizer.fit_transform(texts)
        if tf.shape[0] > 0:
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(tf)
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_features_ind = topic.argsort()[-10:]
                top_features = [feature_names[i] for i in top_features_ind]
                topics.append(" ".join(top_features))
            topic_labels = lda.transform(tf).argmax(axis=1)
            return topics, topic_labels
    except Exception as e:
        app.logger.error(f"Error in topic extraction: {e}")
        return [], []

def get_key_phrases(text):
    if not text.strip():
        return []
    try:
        blob = TextBlob(text)
        return blob.noun_phrases[:5]
    except Exception as e:
        app.logger.error(f"Error in key phrase extraction: {e}")
        return []

def analyze_text_for_actions(text):
    actions = []
    text_lower = text.lower()
    action_triggers = {
        'response time': ['slow response', 'take too long', "didn't respond"],
        'bug': ['bug', 'error', 'crash', 'not working'],
        'feature request': ['wish', 'would like', 'please add', 'need feature'],
        'pricing': ['pricing', 'price', 'too expensive', 'cost too much', 'overpriced', 'price high'],
        'documentation': ['documentation', 'manual', 'hard to understand', 'lack of guidance', 'documentation poor', 'no manual'],
        'ui/ux': ['hard to use', 'confusing interface', 'bad design', 'poor layout']
    }
    for category, patterns in action_triggers.items():
        for pattern in patterns:
            if pattern in text_lower:
                actions.append((category, pattern))
                break
    return actions

def generate_insights(df):
    if df.empty:
        return {}, df
    required_columns = ['text', 'date']
    for col in required_columns:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")
    try:
        df[['sentiment', 'sentiment_score']] = df['text'].apply(lambda x: pd.Series(analyze_sentiment(x)))
        topics, topic_labels = extract_topics(df['text'].tolist())
        if topics:
            df['topic'] = topic_labels
            df['topic_label'] = df['topic'].apply(lambda x: topics[x] if x < len(topics) else "unknown")
        else:
            df['topic_label'] = "unknown"
        df['key_phrases'] = df['text'].apply(get_key_phrases)
        df['actionable_items'] = df['text'].apply(analyze_text_for_actions)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        insights = {
            'sentiment_distribution': dict(df['sentiment'].value_counts()),
            'topic_distribution': dict(df['topic_label'].value_counts()) if 'topic_label' in df.columns else {},
            'top_positive_feedback': df[df['sentiment'] == 'positive']['text'].tolist()[:5],
            'top_negative_feedback': df[df['sentiment'] == 'negative']['text'].tolist()[:5],
            'common_key_phrases': list(Counter([phrase for sublist in df['key_phrases'].tolist() for phrase in sublist]).most_common(10)),
            'actionable_items': list(Counter([item[0] for sublist in df['actionable_items'].tolist() for item in sublist]).most_common(10)),
        }
        if 'date' in df.columns and not df['date'].isnull().all():
            df['month'] = df['date'].dt.to_period('M')
            sentiment_trends = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
            if not sentiment_trends.empty:
                insights['sentiment_over_time'] = [
                    {
                        'date': str(month),
                        'positive': int(row.get('positive', 0)),
                        'negative': int(row.get('negative', 0)),
                        'neutral': int(row.get('neutral', 0))
                    } for month, row in sentiment_trends.reset_index().iterrows()
                ]
        return insights, df
    except Exception as e:
        app.logger.error(f"Error in generating insights: {e}")
        return {}, df
# The rest of the implementation will follow the spec, including routes, model training, and analysis functions.



# Export functions for testing
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
globals()['allowed_file'] = allowed_file

def train_model(df):
    try:
        # Prepare data for training
        # For demo purposes, we'll create dummy labels if they don't exist
        if 'label' not in df.columns:
            # Create dummy labels based on some simple rules for demonstration
            df['label'] = 'neutral'
            df.loc[df['sentiment'] == 'positive', 'label'] = 'positive'
            df.loc[df['sentiment'] == 'negative', 'label'] = 'negative'
            df.loc[df['text'].str.contains('customer service', case=False), 'label'] = 'customer_service'
            df.loc[df['text'].str.contains('price|pricing|cost', case=False), 'label'] = 'pricing'
        # Encode labels
        label_encoder = LabelEncoder()
        df['label_encoded'] = label_encoder.fit_transform(df['label'])
        # Split data
        texts = df['text'].tolist()
        labels = df['label_encoded'].tolist()
        # Split into train and test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        # Create datasets
        class FeedbackDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            def __len__(self):
                return len(self.texts)
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer)
        test_dataset = FeedbackDataset(test_texts, test_labels, tokenizer)
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        # Define Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        # Train the model
        trainer.train()
        # Save the model
        model.save_pretrained('./saved_model')
        tokenizer.save_pretrained('./saved_model')
        # Evaluate
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        report = classification_report(test_labels, preds, target_names=label_encoder.classes_)
        return {
            'success': True,
            'message': 'Model trained and saved successfully.',
            'evaluation_report': report
        }
    except Exception as e:
        app.logger.error(f"Error in training model: {e}")
        return {
            'success': False,
            'error': str(e)
        }
globals()['train_model'] = train_model
