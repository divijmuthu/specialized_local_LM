"""
Data preprocessing pipeline with text cleaning and embedding generation
"""
import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Optional, Tuple
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import spacy
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from config import Config

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            logger.warning("NLTK data download failed, some features may not work")
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic preprocessing")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> str:
        """Complete text preprocessing pipeline"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract text features"""
        if not text:
            return {}
        
        features = {}
        
        # Basic features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Character features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        
        # Sentiment features (basic)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointed', 'poor']
        
        words = text.lower().split()
        features['positive_word_ratio'] = sum(1 for word in words if word in positive_words) / len(words) if words else 0
        features['negative_word_ratio'] = sum(1 for word in words if word in negative_words) / len(words) if words else 0
        
        return features

class EmbeddingGenerator:
    """Generate embeddings for text data"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        if 'distilbert' in model_name.lower():
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._get_batch_embeddings(batch_texts)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def _get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=Config.MAX_SEQUENCE_LENGTH
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = outputs[0].mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def get_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.get_embeddings([text])[0]

class FeatureEngineer:
    """Feature engineering for business data"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
    
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-specific features"""
        df_features = df.copy()
        
        # Customer Lifetime Value
        if 'total_spend' in df.columns and 'retention_rate' in df.columns:
            df_features['CLV'] = df['total_spend'] / (1 - df['retention_rate'])
        
        # Recency, Frequency, Monetary (RFM) Analysis
        if 'last_purchase_date' in df.columns:
            df_features['last_purchase_date'] = pd.to_datetime(df_features['last_purchase_date'])
            df_features['recency'] = (datetime.now() - df_features['last_purchase_date']).dt.days
        
        if 'purchase_count' in df.columns:
            df_features['frequency'] = df['purchase_count']
        
        if 'total_spend' in df.columns:
            df_features['monetary'] = df['total_spend']
        
        # Customer segmentation features
        if all(col in df.columns for col in ['recency', 'frequency', 'monetary']):
            # RFM scores (1-5 scale)
            df_features['recency_score'] = pd.qcut(df_features['recency'], 5, labels=[5,4,3,2,1])
            df_features['frequency_score'] = pd.qcut(df_features['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
            df_features['monetary_score'] = pd.qcut(df_features['monetary'], 5, labels=[1,2,3,4,5])
            
            # RFM segment
            df_features['rfm_segment'] = (
                df_features['recency_score'].astype(str) + 
                df_features['frequency_score'].astype(str) + 
                df_features['monetary_score'].astype(str)
            )
        
        return df_features
    
    def create_sales_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sales-specific features"""
        df_features = df.copy()
        
        # Time-based features
        if 'date' in df.columns:
            df_features['date'] = pd.to_datetime(df_features['date'])
            df_features['year'] = df_features['date'].dt.year
            df_features['month'] = df_features['date'].dt.month
            df_features['quarter'] = df_features['date'].dt.quarter
            df_features['day_of_week'] = df_features['date'].dt.day_name()
            df_features['is_weekend'] = df_features['date'].dt.dayofweek >= 5
        
        # Revenue features
        if 'amount' in df.columns and 'quantity' in df.columns:
            df_features['total_revenue'] = df_features['amount'] * df_features['quantity']
        
        # Price analysis
        if 'amount' in df.columns:
            df_features['price_tier'] = pd.cut(
                df_features['amount'], 
                bins=[0, 50, 100, 200, 1000], 
                labels=['Low', 'Medium', 'High', 'Premium']
            )
        
        # Seasonal features
        if 'month' in df_features.columns:
            df_features['season'] = df_features['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        
        return df_features
    
    def create_feedback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feedback-specific features"""
        df_features = df.copy()
        
        # Sentiment features
        if 'rating' in df.columns:
            df_features['sentiment'] = df_features['rating'].apply(
                lambda x: 'Positive' if x >= 4 else 'Negative' if x <= 2 else 'Neutral'
            )
        
        # Text features
        if 'text' in df.columns:
            preprocessor = TextPreprocessor()
            df_features['text_length'] = df_features['text'].str.len()
            df_features['word_count'] = df_features['text'].str.split().str.len()
            df_features['cleaned_text'] = df_features['text'].apply(preprocessor.preprocess_text)
        
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = set(df_encoded[col].astype(str).unique())
                    known_values = set(self.encoders[col].classes_)
                    new_values = unique_values - known_values
                    
                    if new_values:
                        # Add new values to encoder
                        all_values = list(known_values) + list(new_values)
                        self.encoders[col] = LabelEncoder()
                        self.encoders[col].fit(all_values)
                    
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()
        
        for col in numerical_columns:
            if col in df_scaled.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df_scaled[[col]])
                else:
                    df_scaled[col] = self.scalers[col].transform(df_scaled[[col]])
        
        return df_scaled
    
    def create_tfidf_features(self, texts: List[str], max_features: int = 1000) -> np.ndarray:
        """Create TF-IDF features"""
        if 'tfidf' not in self.vectorizers:
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = self.vectorizers['tfidf'].fit_transform(texts)
        else:
            tfidf_matrix = self.vectorizers['tfidf'].transform(texts)
        
        return tfidf_matrix.toarray()
    
    def save_preprocessors(self, output_dir: str):
        """Save preprocessors for later use"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            with open(os.path.join(output_dir, f'scaler_{name}.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save encoders
        for name, encoder in self.encoders.items():
            with open(os.path.join(output_dir, f'encoder_{name}.pkl'), 'wb') as f:
                pickle.dump(encoder, f)
        
        # Save vectorizers
        for name, vectorizer in self.vectorizers.items():
            with open(os.path.join(output_dir, f'vectorizer_{name}.pkl'), 'wb') as f:
                pickle.dump(vectorizer, f)
    
    def load_preprocessors(self, input_dir: str):
        """Load preprocessors"""
        # Load scalers
        for file in os.listdir(input_dir):
            if file.startswith('scaler_'):
                name = file.replace('scaler_', '').replace('.pkl', '')
                with open(os.path.join(input_dir, file), 'rb') as f:
                    self.scalers[name] = pickle.load(f)
        
        # Load encoders
        for file in os.listdir(input_dir):
            if file.startswith('encoder_'):
                name = file.replace('encoder_', '').replace('.pkl', '')
                with open(os.path.join(input_dir, file), 'rb') as f:
                    self.encoders[name] = pickle.load(f)
        
        # Load vectorizers
        for file in os.listdir(input_dir):
            if file.startswith('vectorizer_'):
                name = file.replace('vectorizer_', '').replace('.pkl', '')
                with open(os.path.join(input_dir, file), 'rb') as f:
                    self.vectorizers[name] = pickle.load(f)

class DataPreprocessor:
    """Main data preprocessing pipeline"""
    
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator()
        self.feature_engineer = FeatureEngineer()
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess all data"""
        logger.info("Starting data preprocessing")
        
        processed_data = {}
        
        for name, df in data.items():
            try:
                logger.info(f"Preprocessing {name}")
                
                # Apply specific preprocessing based on data type
                if name == 'customers':
                    df_processed = self.feature_engineer.create_customer_features(df)
                elif name == 'sales':
                    df_processed = self.feature_engineer.create_sales_features(df)
                elif name == 'feedback':
                    df_processed = self.feature_engineer.create_feedback_features(df)
                else:
                    df_processed = df.copy()
                
                # Add text embeddings if text column exists
                if 'text' in df_processed.columns:
                    logger.info(f"Generating embeddings for {name}")
                    texts = df_processed['text'].fillna('').tolist()
                    embeddings = self.embedding_generator.get_embeddings(texts)
                    
                    # Add embeddings as columns
                    for i in range(embeddings.shape[1]):
                        df_processed[f'embedding_{i}'] = embeddings[:, i]
                
                # Add cleaned text if text column exists
                if 'text' in df_processed.columns:
                    df_processed['cleaned_text'] = df_processed['text'].apply(
                        self.text_preprocessor.preprocess_text
                    )
                
                processed_data[name] = df_processed
                logger.info(f"Completed preprocessing {name}: {len(df_processed)} records")
                
            except Exception as e:
                logger.error(f"Error preprocessing {name}: {e}")
                processed_data[name] = df
        
        return processed_data
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame], output_dir: str):
        """Save processed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data.items():
            # Save as CSV
            csv_path = os.path.join(output_dir, f"{name}_preprocessed.csv")
            df.to_csv(csv_path, index=False)
            
            # Save as parquet for better performance
            parquet_path = os.path.join(output_dir, f"{name}_preprocessed.parquet")
            df.to_parquet(parquet_path, index=False)
            
            logger.info(f"Saved processed {name} to {csv_path} and {parquet_path}")
        
        # Save preprocessors
        preprocessor_dir = os.path.join(output_dir, 'preprocessors')
        self.feature_engineer.save_preprocessors(preprocessor_dir)

if __name__ == "__main__":
    # Test the preprocessing pipeline
    from data_collectors import SampleDataGenerator
    
    # Generate sample data
    generator = SampleDataGenerator()
    sample_data = {
        'customers': generator.generate_customer_data(100),
        'sales': generator.generate_sales_data(500),
        'feedback': generator.generate_feedback_data(200)
    }
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_data(sample_data)
    
    # Save processed data
    preprocessor.save_processed_data(processed_data, './data/processed')
    
    print("Data preprocessing completed successfully!")
