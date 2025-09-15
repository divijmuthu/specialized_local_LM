import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_mining import (
    TemporalPattern,
    MultiScaleTemporalAnalyzer,
    TemporalAdvantageMiner,
    AnomalyPredictor,
    fast_correlation
)

def generate_sample_data():
    timestamps = np.arange(0, 1000)
    data = np.sin(2 * np.pi * timestamps / 100) + np.random.normal(0, 0.1, 1000)
    df = pd.DataFrame({'timestamp': timestamps, 'value': data})
    return {'sample': df}

def test_multiscale_temporal_analyzer_patterns():
    analyzer = MultiScaleTemporalAnalyzer(scales=np.array([1, 2, 4, 8, 16]))
    sample = generate_sample_data()['sample']
    results = analyzer.analyze(sample['value'].values, sample['timestamp'].values)
    assert 'patterns' in results
    assert isinstance(results['patterns'], list)

def test_multiscale_temporal_analyzer_anomalies():
    analyzer = MultiScaleTemporalAnalyzer(scales=np.array([1, 2, 4, 8, 16]))
    sample = generate_sample_data()['sample']
    results = analyzer.analyze(sample['value'].values, sample['timestamp'].values)
    assert 'anomalies' in results
    assert isinstance(results['anomalies'], list)

def test_multiscale_temporal_analyzer_cycles_trends():
    analyzer = MultiScaleTemporalAnalyzer(scales=np.array([1, 2, 4, 8, 16]))
    sample = generate_sample_data()['sample']
    results = analyzer.analyze(sample['value'].values, sample['timestamp'].values)
    assert 'cycles' in results
    assert 'trends' in results
    assert isinstance(results['cycles'], list)
    assert isinstance(results['trends'], list)

def test_fast_correlation():
    x = np.random.randn(100)
    y = np.roll(x, 5)
    corrs = fast_correlation(x, y, max_lag=10)
    assert isinstance(corrs, np.ndarray)
    assert corrs.shape[0] == 21

def test_temporal_advantage_miner():
    # Use smaller scales for testing to avoid long execution time
    small_scales = np.logspace(-1, 2, 10)  # 10 scales instead of 1000
    miner = TemporalAdvantageMiner(config={})
    miner.analyzer.scales = small_scales  # Override with smaller scales
    data = generate_sample_data()
    patterns = miner.analyzer.analyze(data['sample']['value'].values, data['sample']['timestamp'].values)
    assert isinstance(patterns, dict)
    assert 'patterns' in patterns

def test_anomaly_predictor():
    predictor = AnomalyPredictor()
    data = np.random.randn(200)
    anomalies = predictor.predict(data)
    assert isinstance(anomalies, list)
    for anomaly in anomalies:
        assert 'time_until' in anomaly
        assert 'severity' in anomaly
        assert 'confidence' in anomaly
        assert 'expected_value' in anomaly
