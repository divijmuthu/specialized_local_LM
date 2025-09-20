from flask import Flask, jsonify, render_template, request
import pandas as pd
import json
import time
import os

def simulate_data_ingestion(data_sources):
    ingested_data = {}
    for source in data_sources:
        print(f"Connecting to {source['name']}...")
        time.sleep(1)
        if source['type'] == 'csv':
            data = pd.read_csv(source['path'])
        elif source['type'] == 'json':
            with open(source['path'], 'r') as f:
                data = json.load(f)
        else:
            data = f"Sample data from {source['name']}"
        ingested_data[source['name']] = data
        print(f"Successfully ingested data from {source['name']}")
    return ingested_data

def generate_insights(ingested_data):
    insights = []
    if 'Clinical Trials Database' in ingested_data:
        insights.append({
            'description': 'Hidden patient segment in Phase 2 trials with 30% higher efficacy',
            'impact': 2300,
            'data_source': 'Clinical Trials Database'
        })
    if 'Financial Transactions' in ingested_data:
        insights.append({
            'description': 'Fraudulent transactions pattern detected in Southeast region',
            'impact': 1800,
            'data_source': 'Financial Transactions'
        })
    # Always add both insights for demo robustness
    if not any('efficacy' in i['description'] for i in insights):
        insights.append({
            'description': 'Hidden patient segment in Phase 2 trials with 30% higher efficacy',
            'impact': 2300,
            'data_source': 'Clinical Trials Database'
        })
    if not any('fraudulent' in i['description'] for i in insights):
        insights.append({
            'description': 'Fraudulent transactions pattern detected in Southeast region',
            'impact': 1800,
            'data_source': 'Financial Transactions'
        })
    return insights

def simulate_expert_execution(insights):
    actions = []
    for insight in insights:
        if 'higher efficacy' in insight['description'].lower():
            actions.append({
                'description': f'Engage additional patients in {insight["data_source"].lower()} segment',
                'status': 'In Progress',
                'insight': insight['description']
            })
        elif 'fraudulent' in insight['description'].lower():
            actions.append({
                'description': f'Implement real-time fraud detection in {insight["data_source"].lower()} region',
                'status': 'Completed',
                'insight': insight['description']
            })
    return actions

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ingest-data', methods=['POST'])
def ingest_data():
    data_sources = [
        {'name': 'Clinical Trials Database', 'type': 'csv', 'path': 'mock_data/clinical_trials.csv'},
        {'name': 'Financial Transactions', 'type': 'json', 'path': 'mock_data/transactions.json'},
        {'name': 'Customer CRM', 'type': 'csv', 'path': 'mock_data/crm_data.csv'}
    ]
    ingested_data = simulate_data_ingestion(data_sources)
    return jsonify({'status': 'success', 'data': 'Data ingestion simulated'})

@app.route('/generate-insights', methods=['POST'])
def generate_insights_route():
    ingested_data = {
        'Clinical Trials Database': 'Sample data',
        'Financial Transactions': 'Sample data',
        'Customer CRM': 'Sample data'
    }
    insights = generate_insights(ingested_data)
    return jsonify({'status': 'success', 'insights': insights})

@app.route('/execute-actions', methods=['POST'])
def execute_actions():
    insights = [
        {'description': 'Hidden patient segment in Phase 2 trials with 30% higher efficacy', 'impact': 2300, 'data_source': 'Clinical Trials Database'},
        {'description': 'Fraudulent transactions pattern detected in Southeast region', 'impact': 1800, 'data_source': 'Financial Transactions'}
    ]
    actions = simulate_expert_execution(insights)
    return jsonify({'status': 'success', 'actions': actions})

if __name__ == '__main__':
    app.run(debug=True)
