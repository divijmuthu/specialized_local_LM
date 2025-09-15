"""
moccetSmallModel - A specialized local LM for temporal advantage mining and counterfactual analysis
"""

__version__ = "1.0.0"
__author__ = "moccet"

from .temporal_mining import (
    TemporalPattern,
    MultiScaleTemporalAnalyzer,
    TemporalAdvantageMiner,
    AnomalyPredictor,
    fast_correlation
)

from .counterfactual_engine import (
    AlternativeTimeline,
    Decision,
    BusinessSimulator,
    CounterfactualRealityEngine
)

__all__ = [
    'TemporalPattern',
    'MultiScaleTemporalAnalyzer', 
    'TemporalAdvantageMiner',
    'AnomalyPredictor',
    'fast_correlation',
    'AlternativeTimeline',
    'Decision',
    'BusinessSimulator',
    'CounterfactualRealityEngine'
]
