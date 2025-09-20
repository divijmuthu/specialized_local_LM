import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import simulate_data_ingestion, generate_insights, simulate_expert_execution

def test_ingestion_with_missing_source(monkeypatch):
    sources = [
        {'name': 'Unknown Source', 'type': 'csv', 'path': 'mock_data/unknown.csv'}
    ]
    def fake_read_csv(path):
        return [{'mock': 'csv'}]
    monkeypatch.setattr('pandas.read_csv', fake_read_csv)
    class DummyFile:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr('builtins.open', lambda path, mode: DummyFile())
    ingested = simulate_data_ingestion(sources)
    assert 'Unknown Source' in ingested
    assert isinstance(ingested['Unknown Source'], list) or isinstance(ingested['Unknown Source'], str)

def test_generate_insights_with_empty_data():
    ingested_data = {}
    insights = generate_insights(ingested_data)
    # Should still return demo insights for robustness
    assert any('efficacy' in i['description'].lower() for i in insights)
    assert any('fraudulent' in i['description'].lower() for i in insights)

def test_expert_execution_with_unexpected_insight():
    insights = [
        {'description': 'Unexpected pattern found in CRM', 'impact': 1000, 'data_source': 'Customer CRM'}
    ]
    actions = simulate_expert_execution(insights)
    # Should not crash, may return empty or default actions
    assert isinstance(actions, list)
    # If no matching logic, actions may be empty
    assert all('description' in a for a in actions) or actions == []
