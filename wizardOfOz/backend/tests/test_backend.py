import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import simulate_data_ingestion, generate_insights, simulate_expert_execution

class DummySource:
    def __init__(self, name, type_, path):
        self.name = name
        self.type = type_
        self.path = path

def test_simulate_data_ingestion(monkeypatch):
    sources = [
        {'name': 'Clinical Trials Database', 'type': 'csv', 'path': 'mock_data/clinical_trials.csv'},
        {'name': 'Financial Transactions', 'type': 'json', 'path': 'mock_data/transactions.json'},
        {'name': 'Customer CRM', 'type': 'csv', 'path': 'mock_data/crm_data.csv'}
    ]
    def fake_read_csv(path):
        return [{'mock': 'csv'}]
    def fake_json_load(f):
        return [{'mock': 'json'}]
    class DummyFile:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr('pandas.read_csv', fake_read_csv)
    monkeypatch.setattr('json.load', fake_json_load)
    monkeypatch.setattr('builtins.open', lambda path, mode: DummyFile())
    ingested = simulate_data_ingestion(sources)
    assert 'Clinical Trials Database' in ingested
    assert 'Financial Transactions' in ingested
    assert 'Customer CRM' in ingested

def test_generate_insights():
    ingested_data = {
        'Clinical Trials Database': 'Sample data',
        'Financial Transactions': 'Sample data',
        'Customer CRM': 'Sample data'
    }
    insights = generate_insights(ingested_data)
    assert any('efficacy' in i['description'].lower() for i in insights)
    assert any('fraudulent' in i['description'].lower() for i in insights)

def test_simulate_expert_execution():
    insights = [
        {'description': 'Hidden patient segment in Phase 2 trials with 30% higher efficacy', 'impact': 2300, 'data_source': 'Clinical Trials Database'},
        {'description': 'Fraudulent transactions pattern detected in Southeast region', 'impact': 1800, 'data_source': 'Financial Transactions'}
    ]
    actions = simulate_expert_execution(insights)
    assert any('In Progress' == a['status'] for a in actions)
    assert any('Completed' == a['status'] for a in actions)
