"""
Hierarchical Model Architecture for SLM Business Insights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, DistilBertModel, DistilBertConfig
from typing import Dict, List, Optional, Tuple
import logging
from config import Config

logger = logging.getLogger(__name__)

class HighLevelModule(nn.Module):
    """High-level module for strategic insights and patterns"""
    
    def __init__(self, input_size: int, hidden_size: int = 768, num_layers: int = 4):
        super(HighLevelModule, self).__init__()
        
        # Use BERT-based architecture for high-level understanding
        self.bert_config = BertConfig.from_pretrained(
            'bert-base-uncased',
            num_hidden_layers=num_layers,
            hidden_size=hidden_size
        )
        self.bert = BertModel(self.bert_config)
        
        # Additional layers for business insights
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Output layers for different business insights
        self.strategic_classifier = nn.Linear(hidden_size, 10)  # Strategic categories
        self.trend_predictor = nn.Linear(hidden_size, 1)       # Trend prediction
        self.opportunity_detector = nn.Linear(hidden_size, 5)  # Opportunity types
    
    def forward(self, input_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for high-level module"""
        # BERT processing
        outputs = self.bert(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Self-attention for strategic insights
        attended_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        attended_output = self.layer_norm(attended_output + hidden_states)
        attended_output = self.dropout(attended_output)
        
        # Global average pooling
        pooled_output = attended_output.mean(dim=1)
        
        # Generate different types of insights
        strategic_output = self.strategic_classifier(pooled_output)
        trend_output = self.trend_predictor(pooled_output)
        opportunity_output = self.opportunity_detector(pooled_output)
        
        return {
            'strategic_insights': strategic_output,
            'trend_prediction': trend_output,
            'opportunity_detection': opportunity_output,
            'hidden_states': hidden_states,
            'pooled_output': pooled_output
        }

class LowLevelModule(nn.Module):
    """Low-level module for detailed analysis and feature extraction"""
    
    def __init__(self, input_size: int, hidden_size: int = 512):
        super(LowLevelModule, self).__init__()
        
        # Use DistilBERT for efficient low-level processing
        self.distilbert_config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
        self.distilbert = DistilBertModel(self.distilbert_config)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.distilbert_config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Specialized heads for different analysis types
        self.sentiment_head = nn.Linear(hidden_size, 3)      # Positive, Negative, Neutral
        self.category_head = nn.Linear(hidden_size, 20)      # Product categories
        self.priority_head = nn.Linear(hidden_size, 5)       # Priority levels
        self.urgency_head = nn.Linear(hidden_size, 3)        # Urgency levels
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for low-level module"""
        # DistilBERT processing
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Feature extraction
        features = self.feature_extractor(hidden_states.mean(dim=1))
        
        # Generate specialized outputs
        sentiment_output = self.sentiment_head(features)
        category_output = self.category_head(features)
        priority_output = self.priority_head(features)
        urgency_output = self.urgency_head(features)
        
        return {
            'sentiment': sentiment_output,
            'category': category_output,
            'priority': priority_output,
            'urgency': urgency_output,
            'features': features,
            'hidden_states': hidden_states
        }

class HierarchicalModel(nn.Module):
    """Main hierarchical model combining high-level and low-level modules"""
    
    def __init__(self, 
                 vocab_size: int = 30522,
                 embedding_size: int = 768,
                 high_level_hidden_size: int = 768,
                 low_level_hidden_size: int = 512,
                 output_size: int = 100):
        super(HierarchicalModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # High-level and low-level modules
        self.high_level_module = HighLevelModule(embedding_size, high_level_hidden_size)
        self.low_level_module = LowLevelModule(embedding_size, low_level_hidden_size)
        
        # Integration layer
        self.integration_layer = nn.Sequential(
            nn.Linear(high_level_hidden_size + low_level_hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_size)
        )
        
        # Business insight heads
        self.revenue_predictor = nn.Linear(output_size, 1)
        self.cost_optimizer = nn.Linear(output_size, 1)
        self.customer_segmenter = nn.Linear(output_size, 10)
        self.risk_assessor = nn.Linear(output_size, 5)
        self.opportunity_ranker = nn.Linear(output_size, 1)
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass for hierarchical model"""
        
        # Get embeddings
        input_embeddings = self.embedding(input_ids)
        
        # Low-level processing
        low_level_outputs = self.low_level_module(input_ids, attention_mask)
        
        # High-level processing (using low-level features as input)
        high_level_outputs = self.high_level_module(
            low_level_outputs['hidden_states'], 
            attention_mask
        )
        
        # Integrate high-level and low-level features
        combined_features = torch.cat([
            high_level_outputs['pooled_output'],
            low_level_outputs['features']
        ], dim=1)
        
        # Generate integrated representation
        integrated_output = self.integration_layer(combined_features)
        
        # Generate business insights
        revenue_prediction = self.revenue_predictor(integrated_output)
        cost_optimization = self.cost_optimizer(integrated_output)
        customer_segmentation = self.customer_segmenter(integrated_output)
        risk_assessment = self.risk_assessor(integrated_output)
        opportunity_ranking = self.opportunity_ranker(integrated_output)
        
        outputs = {
            'revenue_prediction': revenue_prediction,
            'cost_optimization': cost_optimization,
            'customer_segmentation': customer_segmentation,
            'risk_assessment': risk_assessment,
            'opportunity_ranking': opportunity_ranking,
            'integrated_features': integrated_output
        }
        
        if return_components:
            outputs.update({
                'high_level_outputs': high_level_outputs,
                'low_level_outputs': low_level_outputs
            })
        
        return outputs

class BusinessInsightHead(nn.Module):
    """Specialized head for specific business insights"""
    
    def __init__(self, input_size: int, insight_type: str):
        super(BusinessInsightHead, self).__init__()
        self.insight_type = insight_type
        
        if insight_type == 'customer_lifetime_value':
            self.head = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
        elif insight_type == 'churn_prediction':
            self.head = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 2)  # Churn/No Churn
            )
        elif insight_type == 'price_optimization':
            self.head = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
        elif insight_type == 'demand_forecasting':
            self.head = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
        else:
            # Generic head
            self.head = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 10)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class MultiTaskHierarchicalModel(nn.Module):
    """Multi-task hierarchical model for various business insights"""
    
    def __init__(self, 
                 vocab_size: int = 30522,
                 embedding_size: int = 768,
                 high_level_hidden_size: int = 768,
                 low_level_hidden_size: int = 512):
        super(MultiTaskHierarchicalModel, self).__init__()
        
        # Base hierarchical model
        self.base_model = HierarchicalModel(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            high_level_hidden_size=high_level_hidden_size,
            low_level_hidden_size=low_level_hidden_size,
            output_size=512
        )
        
        # Specialized heads for different business tasks
        self.insight_heads = nn.ModuleDict({
            'customer_lifetime_value': BusinessInsightHead(512, 'customer_lifetime_value'),
            'churn_prediction': BusinessInsightHead(512, 'churn_prediction'),
            'price_optimization': BusinessInsightHead(512, 'price_optimization'),
            'demand_forecasting': BusinessInsightHead(512, 'demand_forecasting'),
            'sentiment_analysis': BusinessInsightHead(512, 'sentiment_analysis'),
            'market_segmentation': BusinessInsightHead(512, 'market_segmentation'),
            'risk_assessment': BusinessInsightHead(512, 'risk_assessment'),
            'opportunity_detection': BusinessInsightHead(512, 'opportunity_detection')
        })
        
        # Task weights for multi-task learning
        self.task_weights = nn.Parameter(torch.ones(len(self.insight_heads)))
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                task: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task model"""
        
        # Get base model outputs
        base_outputs = self.base_model(input_ids, attention_mask)
        integrated_features = base_outputs['integrated_features']
        
        # Generate insights for all tasks or specific task
        if task is not None:
            if task in self.insight_heads:
                outputs = {task: self.insight_heads[task](integrated_features)}
            else:
                raise ValueError(f"Unknown task: {task}")
        else:
            # Generate all insights
            outputs = {}
            for task_name, head in self.insight_heads.items():
                outputs[task_name] = head(integrated_features)
        
        # Add base outputs
        outputs.update(base_outputs)
        
        return outputs
    
    def get_task_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor], 
                     task: str) -> torch.Tensor:
        """Calculate loss for specific task"""
        if task not in outputs or task not in targets:
            return torch.tensor(0.0, device=outputs[list(outputs.keys())[0]].device)
        
        if task in ['churn_prediction', 'sentiment_analysis']:
            # Classification tasks
            criterion = nn.CrossEntropyLoss()
            return criterion(outputs[task], targets[task])
        else:
            # Regression tasks
            criterion = nn.MSELoss()
            return criterion(outputs[task].squeeze(), targets[task].squeeze())
    
    def get_total_loss(self, outputs: Dict[str, torch.Tensor], 
                      targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate total multi-task loss"""
        total_loss = 0.0
        task_count = 0
        
        for task in self.insight_heads.keys():
            if task in outputs and task in targets:
                task_loss = self.get_task_loss(outputs, targets, task)
                weight = self.task_weights[task_count]
                total_loss += weight * task_loss
                task_count += 1
        
        return total_loss / max(task_count, 1)

# Model factory function
def create_hierarchical_model(model_type: str = 'base', **kwargs) -> nn.Module:
    """Factory function to create different types of hierarchical models"""
    
    if model_type == 'base':
        return HierarchicalModel(**kwargs)
    elif model_type == 'multi_task':
        return MultiTaskHierarchicalModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Model configuration
MODEL_CONFIGS = {
    'small': {
        'vocab_size': 30522,
        'embedding_size': 256,
        'high_level_hidden_size': 256,
        'low_level_hidden_size': 128,
        'output_size': 64
    },
    'medium': {
        'vocab_size': 30522,
        'embedding_size': 512,
        'high_level_hidden_size': 512,
        'low_level_hidden_size': 256,
        'output_size': 256
    },
    'large': {
        'vocab_size': 30522,
        'embedding_size': 768,
        'high_level_hidden_size': 768,
        'low_level_hidden_size': 512,
        'output_size': 512
    }
}

if __name__ == "__main__":
    # Test the hierarchical model
    model = create_hierarchical_model('multi_task', **MODEL_CONFIGS['medium'])
    
    # Create sample input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    outputs = model(input_ids, attention_mask)
    
    print("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    print("Hierarchical model test completed successfully!")
