import pytest
from datetime import datetime, timedelta
from moccetSmallModel.counterfactual_engine import (
    AlternativeTimeline,
    Decision,
    BusinessSimulator,
    CounterfactualRealityEngine
)

def test_decision_and_timeline_dataclasses():
    decision = Decision(
        decision_id="D001",
        timestamp=datetime.now(),
        type="pricing",
        parameters={"price": 100},
        actual_outcome=50000.0,
        alternatives=[]
    )
    timeline = AlternativeTimeline(
        timeline_id="ALT_00001",
        divergence_point=datetime.now(),
        changes_made=[{"price": 110}],
        outcomes={"profit": 120000.0},
        probability=0.5,
        improvement=0.2
    )
    assert isinstance(decision, Decision)
    assert isinstance(timeline, AlternativeTimeline)
    assert timeline.improvement == 0.2

def test_business_simulator_simulate_timeline():
    simulator = BusinessSimulator()
    initial_state = np.random.randn(256)
    decisions = [
        Decision(
            decision_id="D001",
            timestamp=datetime.now(),
            type="pricing",
            parameters={"price": 100},
            actual_outcome=None,
            alternatives=[]
        )
    ]
    result = simulator.simulate_timeline(initial_state, decisions, horizon=10)
    assert "trajectory" in result
    assert "outcomes" in result
    assert result["trajectory"].shape[0] == 11
    assert result["outcomes"].shape[0] == 10

def test_counterfactual_engine_identify_decision_points():
    engine = CounterfactualRealityEngine()
    decisions = [
        Decision(
            decision_id=f"D{i:03d}",
            timestamp=datetime.now() - timedelta(days=i*10),
            type="pricing" if i % 2 == 0 else "marketing",
            parameters={"price": 100 + i},
            actual_outcome=100000 + i*1000,
            alternatives=[]
        ) for i in range(5)
    ]
    critical = engine._identify_decision_points(decisions)
    assert isinstance(critical, list)
    assert all(isinstance(d, Decision) for d in critical)
    assert len(critical) > 0

def test_counterfactual_engine_generate_alternatives():
    engine = CounterfactualRealityEngine()
    decisions = [
        Decision(
            decision_id=f"D{i:03d}",
            timestamp=datetime.now(),
            type="pricing",
            parameters={"price": 100 + i},
            actual_outcome=None,
            alternatives=[]
        ) for i in range(3)
    ]
    alternatives = engine._generate_alternatives(decisions, 5)
    assert isinstance(alternatives, list)
    assert len(alternatives) >= 5
    assert all(isinstance(alt, list) for alt in alternatives)
    assert all(isinstance(d, Decision) for alt in alternatives for d in alt)

def test_counterfactual_engine_simulate_timelines():
    engine = CounterfactualRealityEngine()
    decisions = [
        Decision(
            decision_id=f"D{i:03d}",
            timestamp=datetime.now(),
            type="pricing",
            parameters={"price": 100 + i},
            actual_outcome=None,
            alternatives=[]
        ) for i in range(2)
    ]
    alternatives = engine._generate_alternatives(decisions, 2)
    business_state = {"revenue": 1e6, "profit": 2e5}
    import asyncio
    timelines = asyncio.run(engine._simulate_timelines(business_state, alternatives))
    assert isinstance(timelines, list)
    assert all(isinstance(t, AlternativeTimeline) for t in timelines)
    assert len(timelines) > 0

def test_counterfactual_engine_evaluate_and_find_optimal():
    engine = CounterfactualRealityEngine()
    timelines = [
        AlternativeTimeline(
            timeline_id=f"ALT_{i:05d}",
            divergence_point=datetime.now(),
            changes_made=[{"price": 100 + i}],
            outcomes={"profit": 100000 + i*10000},
            probability=0.5,
            improvement=0.1 * i
        ) for i in range(5)
    ]
    evaluated = engine._evaluate_timelines(timelines, {})
    optimal = engine._find_optimal_timeline(timelines)
    assert isinstance(evaluated, list)
    assert evaluated[0].improvement >= evaluated[-1].improvement
    assert isinstance(optimal, AlternativeTimeline)
    assert optimal.improvement == max(t.improvement for t in timelines)
import numpy as np
import pandas as pd
from moccetSmallModel.counterfactual_engine import CounterfactualRealityEngine, AlternativeTimeline


def test_explore_parallel_realities():
    engine = CounterfactualRealityEngine()
    business_state = {"revenue": 1e6, "profit": 2e5}
    decisions = [
        Decision(
            decision_id=f"D{i:03d}",
            timestamp=datetime.now(),
            type="pricing",
            parameters={"price": 100 + i},
            actual_outcome=None,
            alternatives=[]
        ) for i in range(3)
    ]
    import asyncio
    timelines = asyncio.run(engine.explore_parallel_realities(business_state, decisions, num_realities=5))
    assert isinstance(timelines, list)
    assert len(timelines) > 0
    for timeline in timelines:
        assert isinstance(timeline, AlternativeTimeline)
        assert isinstance(timeline.changes_made, list)
        assert isinstance(timeline.outcomes, dict)
        assert 0.01 <= timeline.probability <= 0.99
