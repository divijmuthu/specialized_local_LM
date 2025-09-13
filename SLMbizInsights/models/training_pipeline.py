"""
Training pipeline with deep supervision for hierarchical models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import get_linear_schedule_with_warmup
from config import Config

logger = logging.getLogger(__name__)

class BusinessDataset(Dataset):
    """Custom dataset for business data"""
    
    def __init__(self, data: Dict[str, Any], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure all arrays have the same length
        self.length = len(data[list(data.keys())[0]])
        for key, value in data.items():
            if len(value) != self.length:
                raise ValueError(f"All data arrays must have the same length. {key} has length {len(value)}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        item = {}
        for key, value in self.data.items():
            if key == 'text':
                # Tokenize text
                encoding = self.tokenizer(
                    str(value[idx]),
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                item['input_ids'] = encoding['input_ids'].squeeze()
                item['attention_mask'] = encoding['attention_mask'].squeeze()
            else:
                item[key] = torch.tensor(value[idx], dtype=torch.float32)
        
        return item

class DeepSupervisionTrainer:
    """Trainer with deep supervision for hierarchical models"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 warmup_steps: int = 100):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions for different tasks
        self.loss_functions = {
            'classification': nn.CrossEntropyLoss(),
            'regression': nn.MSELoss(),
            'binary_classification': nn.BCEWithLogitsLoss()
        }
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def prepare_data(self, 
                    data: Dict[str, pd.DataFrame],
                    test_size: float = 0.2,
                    val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training"""
        
        # Combine all data
        combined_data = {}
        
        # Process text data
        if 'feedback' in data and 'text' in data['feedback'].columns:
            combined_data['text'] = data['feedback']['text'].fillna('').tolist()
            combined_data['sentiment'] = data['feedback']['rating'].values
        
        # Process customer data
        if 'customers' in data:
            customer_data = data['customers']
            if 'CLV' in customer_data.columns:
                combined_data['customer_lifetime_value'] = customer_data['CLV'].values
            if 'segment' in customer_data.columns:
                # Encode segments
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                combined_data['customer_segment'] = le.fit_transform(customer_data['segment'].astype(str))
        
        # Process sales data
        if 'sales' in data:
            sales_data = data['sales']
            if 'total_revenue' in sales_data.columns:
                combined_data['revenue'] = sales_data['total_revenue'].values
        
        # Convert to numpy arrays and ensure same length
        min_length = min(len(v) for v in combined_data.values())
        for key in combined_data:
            combined_data[key] = combined_data[key][:min_length]
        
        # Split data
        indices = np.arange(min_length)
        train_idx, temp_idx = train_test_split(indices, test_size=test_size + val_size, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size/(test_size + val_size), random_state=42)
        
        # Create datasets
        train_data = {key: [value[i] for i in train_idx] for key, value in combined_data.items()}
        val_data = {key: [value[i] for i in val_idx] for key, value in combined_data.items()}
        test_data = {key: [value[i] for i in test_idx] for key, value in combined_data.items()}
        
        # Create data loaders
        train_dataset = BusinessDataset(train_data, self.tokenizer)
        val_dataset = BusinessDataset(val_data, self.tokenizer)
        test_dataset = BusinessDataset(test_data, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with deep supervision"""
        self.model.train()
        total_loss = 0.0
        task_losses = {}
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Calculate losses for different tasks
            total_batch_loss = 0.0
            
            # Sentiment analysis loss
            if 'sentiment' in batch and 'sentiment_analysis' in outputs:
                sentiment_loss = self.loss_functions['classification'](
                    outputs['sentiment_analysis'], 
                    batch['sentiment'].long()
                )
                total_batch_loss += sentiment_loss
                task_losses['sentiment'] = task_losses.get('sentiment', 0) + sentiment_loss.item()
            
            # Customer lifetime value loss
            if 'customer_lifetime_value' in batch and 'customer_lifetime_value' in outputs:
                clv_loss = self.loss_functions['regression'](
                    outputs['customer_lifetime_value'].squeeze(),
                    batch['customer_lifetime_value']
                )
                total_batch_loss += clv_loss
                task_losses['clv'] = task_losses.get('clv', 0) + clv_loss.item()
            
            # Revenue prediction loss
            if 'revenue' in batch and 'revenue_prediction' in outputs:
                revenue_loss = self.loss_functions['regression'](
                    outputs['revenue_prediction'].squeeze(),
                    batch['revenue']
                )
                total_batch_loss += revenue_loss
                task_losses['revenue'] = task_losses.get('revenue', 0) + revenue_loss.item()
            
            # Customer segmentation loss
            if 'customer_segment' in batch and 'customer_segmentation' in outputs:
                segment_loss = self.loss_functions['classification'](
                    outputs['customer_segmentation'],
                    batch['customer_segment'].long()
                )
                total_batch_loss += segment_loss
                task_losses['segmentation'] = task_losses.get('segmentation', 0) + segment_loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
        
        return {
            'total_loss': avg_loss,
            **avg_task_losses
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        task_losses = {}
        predictions = {}
        targets = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Calculate losses
                batch_loss = 0.0
                
                # Sentiment analysis
                if 'sentiment' in batch and 'sentiment_analysis' in outputs:
                    sentiment_loss = self.loss_functions['classification'](
                        outputs['sentiment_analysis'], 
                        batch['sentiment'].long()
                    )
                    batch_loss += sentiment_loss
                    task_losses['sentiment'] = task_losses.get('sentiment', 0) + sentiment_loss.item()
                    
                    # Store predictions for metrics
                    pred = torch.argmax(outputs['sentiment_analysis'], dim=1)
                    predictions['sentiment'] = predictions.get('sentiment', []) + pred.cpu().numpy().tolist()
                    targets['sentiment'] = targets.get('sentiment', []) + batch['sentiment'].cpu().numpy().tolist()
                
                # Customer lifetime value
                if 'customer_lifetime_value' in batch and 'customer_lifetime_value' in outputs:
                    clv_loss = self.loss_functions['regression'](
                        outputs['customer_lifetime_value'].squeeze(),
                        batch['customer_lifetime_value']
                    )
                    batch_loss += clv_loss
                    task_losses['clv'] = task_losses.get('clv', 0) + clv_loss.item()
                
                # Revenue prediction
                if 'revenue' in batch and 'revenue_prediction' in outputs:
                    revenue_loss = self.loss_functions['regression'](
                        outputs['revenue_prediction'].squeeze(),
                        batch['revenue']
                    )
                    batch_loss += revenue_loss
                    task_losses['revenue'] = task_losses.get('revenue', 0) + revenue_loss.item()
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        # Calculate metrics
        metrics = {}
        if 'sentiment' in predictions:
            metrics['sentiment_accuracy'] = accuracy_score(targets['sentiment'], predictions['sentiment'])
            metrics['sentiment_f1'] = f1_score(targets['sentiment'], predictions['sentiment'], average='weighted')
        
        avg_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
        
        return {
            'total_loss': avg_loss,
            **avg_task_losses,
            **metrics
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 10,
              patience: int = 5,
              save_best: bool = True) -> Dict[str, List[float]]:
        """Train the model with early stopping"""
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['train_metrics'].append(train_metrics)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            self.training_history['val_loss'].append(val_metrics['total_loss'])
            self.training_history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            if 'sentiment_accuracy' in val_metrics:
                logger.info(f"Val Sentiment Accuracy: {val_metrics['sentiment_accuracy']:.4f}")
            
            # Early stopping
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if save_best:
                    self.save_model(f"best_model_epoch_{epoch + 1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.training_history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data"""
        self.model.eval()
        metrics = {}
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Calculate metrics for each task
                if 'sentiment' in batch and 'sentiment_analysis' in outputs:
                    pred = torch.argmax(outputs['sentiment_analysis'], dim=1)
                    metrics['sentiment_accuracy'] = accuracy_score(batch['sentiment'].cpu(), pred.cpu())
                    metrics['sentiment_f1'] = f1_score(batch['sentiment'].cpu(), pred.cpu(), average='weighted')
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model and training history"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and training history"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Task-specific losses
        if self.training_history['train_metrics']:
            train_metrics = self.training_history['train_metrics']
            val_metrics = self.training_history['val_metrics']
            
            # Plot task losses
            task_names = [k for k in train_metrics[0].keys() if k != 'total_loss']
            for task in task_names:
                train_task_losses = [m.get(task, 0) for m in train_metrics]
                val_task_losses = [m.get(task, 0) for m in val_metrics]
                
                axes[0, 1].plot(train_task_losses, label=f'Train {task}')
                axes[0, 1].plot(val_task_losses, label=f'Val {task}')
            
            axes[0, 1].set_title('Task-Specific Losses')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Metrics
        if self.training_history['val_metrics']:
            val_metrics = self.training_history['val_metrics']
            metric_names = [k for k in val_metrics[0].keys() if 'accuracy' in k or 'f1' in k]
            
            for metric in metric_names:
                metric_values = [m.get(metric, 0) for m in val_metrics]
                axes[1, 0].plot(metric_values, label=metric)
            
            axes[1, 0].set_title('Validation Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, 'Learning Rate Schedule\n(Not implemented)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Learning Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

class ModelTrainer:
    """Main model trainer class"""
    
    def __init__(self, model_type: str = 'multi_task', model_size: str = 'medium'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model_size = model_size
        
        # Create model
        from .hierarchical_model import create_hierarchical_model, MODEL_CONFIGS
        self.model = create_hierarchical_model(
            model_type, 
            **MODEL_CONFIGS[model_size]
        )
        
        # Initialize tokenizer
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize trainer
        self.trainer = DeepSupervisionTrainer(self.model, self.device)
        self.trainer.tokenizer = self.tokenizer
    
    def train_model(self, 
                   data: Dict[str, pd.DataFrame],
                   num_epochs: int = 10,
                   test_size: float = 0.2,
                   val_size: float = 0.1) -> Dict[str, Any]:
        """Train the model"""
        
        logger.info("Starting model training")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.trainer.prepare_data(
            data, test_size, val_size
        )
        
        # Train model
        training_history = self.trainer.train(
            train_loader, val_loader, num_epochs
        )
        
        # Evaluate on test set
        test_metrics = self.trainer.evaluate(test_loader)
        
        # Save final model
        model_path = os.path.join(Config.TRAINED_MODELS_DIR, f"{self.model_type}_{self.model_size}_final.pt")
        self.trainer.save_model(model_path)
        
        # Plot training history
        plot_path = os.path.join(Config.TRAINED_MODELS_DIR, f"{self.model_type}_{self.model_size}_training_history.png")
        self.trainer.plot_training_history(plot_path)
        
        logger.info("Model training completed")
        
        return {
            'training_history': training_history,
            'test_metrics': test_metrics,
            'model_path': model_path
        }

if __name__ == "__main__":
    # Test the training pipeline
    from pipeline.data_collectors import SampleDataGenerator
    from pipeline.data_preprocessing import DataPreprocessor
    
    # Generate and preprocess sample data
    generator = SampleDataGenerator()
    sample_data = {
        'customers': generator.generate_customer_data(1000),
        'sales': generator.generate_sales_data(2000),
        'feedback': generator.generate_feedback_data(1000)
    }
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_data(sample_data)
    
    # Train model
    trainer = ModelTrainer('multi_task', 'small')
    results = trainer.train_model(processed_data, num_epochs=2)
    
    print("Training completed successfully!")
    print(f"Test metrics: {results['test_metrics']}")
