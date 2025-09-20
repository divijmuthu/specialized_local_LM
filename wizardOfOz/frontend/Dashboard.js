import React, { useEffect } from 'react';
import Chart from 'chart.js/auto';
import 'bootstrap/dist/css/bootstrap.min.css';

const Dashboard = () => {
  useEffect(() => {
    const ctx = document.getElementById('metrics-chart').getContext('2d');
    window.metricsChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Insights', 'Actions', 'ROI'],
        datasets: [{
          label: 'Metrics',
          data: [0, 0, 0],
          backgroundColor: [
            'rgba(255, 99, 132, 0.2)',
            'rgba(54, 162, 235, 0.2)',
            'rgba(75, 192, 192, 0.2)'
          ],
          borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(75, 192, 192, 1)'
          ],
          borderWidth: 1
        }]
      },
      options: {
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
    simulateDemo();
  }, []);

  function updateMetrics(insights, actions, roi) {
    window.metricsChart.data.datasets[0].data = [insights, actions, roi];
    window.metricsChart.update();
    document.getElementById('total-insights').textContent = insights;
    document.getElementById('revenue-impact').textContent = roi * 1000;
  }

  function addDataSource(name, status) {
    const list = document.getElementById('data-sources');
    const item = document.createElement('li');
    item.className = 'list-group-item';
    item.textContent = `${name} (${status})`;
    list.appendChild(item);
  }

  function addInsight(description, impact) {
    const list = document.getElementById('insights-list');
    const item = document.createElement('div');
    item.className = 'list-group-item';
    item.innerHTML = `<h6>Insight</h6><p>${description}</p><p><strong>Potential Impact:</strong> $${impact}K</p>`;
    list.appendChild(item);
  }

  function addAction(description, status) {
    const list = document.getElementById('actions-list');
    const item = document.createElement('div');
    item.className = 'list-group-item';
    item.innerHTML = `<h6>Action</h6><p>${description}</p><p><strong>Status:</strong> ${status}</p>`;
    list.appendChild(item);
  }

  function simulateDemo() {
    addDataSource('Clinical Trials Database', 'Connected');
    addDataSource('Financial Transactions', 'Connected');
    addDataSource('Customer CRM', 'Connected');
    updateMetrics(0, 0, 0);
    setTimeout(() => {
      addInsight('Hidden patient segment in Phase 2 trials with 30% higher efficacy', 2300);
      addInsight('Fraudulent transactions pattern detected in Southeast region', 1800);
      updateMetrics(2, 0, 4100);
    }, 2000);
    setTimeout(() => {
      addAction('Engage additional 200 patients in Phase 2 segment', 'In Progress');
      addAction('Implement real-time fraud detection in Southeast', 'Completed');
      updateMetrics(2, 2, 4100);
    }, 4000);
  }

  return (
    <div className="container-fluid">
      <div className="dashboard" style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', padding: '20px'}}>
        {/* Dashboard Overview */}
        <div className="card">
          <div className="card-header">Dashboard Overview</div>
          <div className="card-body">
            <h5 className="card-title">Key Metrics</h5>
            <p className="card-text">Total Insights Discovered: <span id="total-insights">0</span></p>
            <p className="card-text">Potential Revenue Impact: $<span id="revenue-impact">0</span></p>
            <canvas id="metrics-chart" width="400" height="200"></canvas>
          </div>
        </div>
        {/* Data Sources */}
        <div className="card">
          <div className="card-header">Data Sources</div>
          <div className="card-body">
            <h5 className="card-title">Connected Sources</h5>
            <ul id="data-sources" className="list-group"></ul>
          </div>
        </div>
        {/* Insights Engine */}
        <div className="card">
          <div className="card-header">Insights Engine</div>
          <div className="card-body">
            <h5 className="card-title">Discovered Insights</h5>
            <div id="insights-list" className="list-group"></div>
          </div>
        </div>
        {/* Expert Execution */}
        <div className="card">
          <div className="card-header">Expert Execution</div>
          <div className="card-body">
            <h5 className="card-title">Actions Taken</h5>
            <div id="actions-list" className="list-group"></div>
          </div>
        </div>
        {/* Security and Compliance */}
        <div className="card">
          <div className="card-header">Security and Compliance</div>
          <div className="card-body">
            <h5 className="card-title">Certifications</h5>
            <div className="d-flex justify-content-around">
              <span className="badge badge-secondary">SOC 2 Type 2</span>
              <span className="badge badge-secondary">HIPAA</span>
              <span className="badge badge-secondary">GDPR</span>
            </div>
            <p className="card-text mt-3">All data is encrypted with military-grade encryption and models forget after each session.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
