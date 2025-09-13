"""
Dashboard UI for non-technical users using Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Any

# Configure Streamlit page
st.set_page_config(
    page_title="SLM Business Insights Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .insight-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .recommendation-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class BusinessInsightsDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:5000"
        self.sample_data = None
    
    def load_sample_data(self):
        """Load sample data from API"""
        try:
            response = requests.get(f"{self.api_base_url}/api/data/sample")
            if response.status_code == 200:
                self.sample_data = response.json()
                return True
            else:
                st.error("Failed to load sample data from API")
                return False
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Please ensure the API server is running.")
            return False
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            return False
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🚀 SLM Business Insights Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with navigation"""
        st.sidebar.title("📊 Navigation")
        
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Customer Insights", "Sales Analytics", "Pricing Optimization", 
             "Recommendations", "Data Management", "Settings"]
        )
        
        st.sidebar.markdown("---")
        
        # API Status
        st.sidebar.subheader("🔗 API Status")
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                st.sidebar.success("✅ API Connected")
            else:
                st.sidebar.error("❌ API Error")
        except:
            st.sidebar.error("❌ API Offline")
        
        # Data Status
        st.sidebar.subheader("📁 Data Status")
        if self.sample_data:
            st.sidebar.success("✅ Sample Data Loaded")
            st.sidebar.info(f"Customers: {len(self.sample_data.get('customers', []))}")
            st.sidebar.info(f"Sales: {len(self.sample_data.get('sales', []))}")
            st.sidebar.info(f"Feedback: {len(self.sample_data.get('feedback', []))}")
        else:
            st.sidebar.warning("⚠️ No Data Loaded")
        
        return page
    
    def render_overview(self):
        """Render overview page"""
        st.header("📈 Business Overview")
        
        if not self.sample_data:
            st.warning("Please load sample data first.")
            return
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(self.sample_data.get('customers', []))
            st.metric("Total Customers", total_customers)
        
        with col2:
            total_sales = len(self.sample_data.get('sales', []))
            st.metric("Total Transactions", total_sales)
        
        with col3:
            avg_rating = np.mean([f.get('rating', 0) for f in self.sample_data.get('feedback', [])])
            st.metric("Average Rating", f"{avg_rating:.1f}/5")
        
        with col4:
            total_revenue = sum([s.get('amount', 0) * s.get('quantity', 0) for s in self.sample_data.get('sales', [])])
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sales Trend")
            sales_df = pd.DataFrame(self.sample_data.get('sales', []))
            if not sales_df.empty and 'date' in sales_df.columns:
                sales_df['date'] = pd.to_datetime(sales_df['date'])
                sales_df['revenue'] = sales_df['amount'] * sales_df['quantity']
                daily_sales = sales_df.groupby('date')['revenue'].sum().reset_index()
                
                fig = px.line(daily_sales, x='date', y='revenue', title='Daily Revenue Trend')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("👥 Customer Segments")
            customers_df = pd.DataFrame(self.sample_data.get('customers', []))
            if not customers_df.empty and 'segment' in customers_df.columns:
                segment_counts = customers_df['segment'].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index, title='Customer Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    def render_customer_insights(self):
        """Render customer insights page"""
        st.header("👥 Customer Insights")
        
        if not self.sample_data:
            st.warning("Please load sample data first.")
            return
        
        # Customer Segmentation
        st.subheader("🎯 Customer Segmentation")
        
        if st.button("Run Customer Segmentation", key="segmentation_btn"):
            with st.spinner("Analyzing customer segments..."):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/api/insights/customer-segmentation",
                        json={
                            "customers": self.sample_data.get('customers', [])[:100],
                            "n_clusters": 5
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display segments
                        segments_df = pd.DataFrame({
                            'customer_id': range(len(data['segments'])),
                            'segment': data['segments']
                        })
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Segment Distribution")
                            segment_counts = segments_df['segment'].value_counts()
                            fig = px.bar(x=segment_counts.index, y=segment_counts.values, 
                                       title="Customer Segments")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Segment Analysis")
                            for segment, analysis in data['segment_analysis'].items():
                                st.markdown(f"""
                                <div class="insight-card">
                                    <h4>{segment.replace('_', ' ').title()}</h4>
                                    <p><strong>Size:</strong> {analysis['size']} customers ({analysis['percentage']:.1f}%)</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to run customer segmentation")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Sentiment Analysis
        st.subheader("😊 Sentiment Analysis")
        
        if st.button("Analyze Customer Sentiment", key="sentiment_btn"):
            with st.spinner("Analyzing sentiment..."):
                try:
                    feedback_texts = [f.get('text', '') for f in self.sample_data.get('feedback', [])[:20]]
                    
                    response = requests.post(
                        f"{self.api_base_url}/api/insights/sentiment-analysis",
                        json={"texts": feedback_texts}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Sentiment Distribution")
                            summary = data['summary']
                            fig = px.pie(
                                values=[summary['positive_count'], summary['negative_count'], summary['neutral_count']],
                                names=['Positive', 'Negative', 'Neutral'],
                                title="Customer Sentiment"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Sentiment Summary")
                            st.markdown(f"""
                            <div class="insight-card">
                                <p><strong>Total Reviews:</strong> {summary['total_texts']}</p>
                                <p><strong>Positive:</strong> {summary['positive_count']}</p>
                                <p><strong>Negative:</strong> {summary['negative_count']}</p>
                                <p><strong>Neutral:</strong> {summary['neutral_count']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to analyze sentiment")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Churn Prediction
        st.subheader("⚠️ Churn Prediction")
        
        if st.button("Predict Customer Churn", key="churn_btn"):
            with st.spinner("Predicting churn..."):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/api/insights/churn-prediction",
                        json={"customer_data": self.sample_data.get('customers', [])[:50]}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display churn predictions
                        churn_df = pd.DataFrame(data['predictions'])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Churn Risk Distribution")
                            risk_counts = churn_df['risk_level'].value_counts()
                            fig = px.bar(x=risk_counts.index, y=risk_counts.values, 
                                       title="Customer Churn Risk")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("High Risk Customers")
                            high_risk = churn_df[churn_df['risk_level'] == 'high'].head(10)
                            if not high_risk.empty:
                                st.dataframe(high_risk[['customer_id', 'churn_probability', 'risk_level']])
                            else:
                                st.info("No high-risk customers found")
                        
                        # Summary
                        summary = data['summary']
                        st.markdown(f"""
                        <div class="insight-card">
                            <h4>Churn Prediction Summary</h4>
                            <p><strong>Total Customers:</strong> {summary['total_customers']}</p>
                            <p><strong>High Risk:</strong> {summary['high_risk']}</p>
                            <p><strong>Medium Risk:</strong> {summary['medium_risk']}</p>
                            <p><strong>Low Risk:</strong> {summary['low_risk']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to predict churn")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    def render_sales_analytics(self):
        """Render sales analytics page"""
        st.header("📊 Sales Analytics")
        
        if not self.sample_data:
            st.warning("Please load sample data first.")
            return
        
        # Sales Forecasting
        st.subheader("📈 Sales Forecasting")
        
        if st.button("Generate Sales Forecast", key="forecast_btn"):
            with st.spinner("Generating sales forecast..."):
                try:
                    sales_data = self.sample_data.get('sales', [])[:50]
                    # Convert to proper format
                    forecast_data = []
                    for sale in sales_data:
                        forecast_data.append({
                            'date': sale.get('date', '2023-01-01'),
                            'sales': sale.get('amount', 0) * sale.get('quantity', 0)
                        })
                    
                    response = requests.post(
                        f"{self.api_base_url}/api/insights/sales-forecasting",
                        json={"sales_data": forecast_data, "periods": 12}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display forecast
                        forecast_df = pd.DataFrame(data['forecast'])
                        historical_df = pd.DataFrame(list(data['historical_data'].items()), 
                                                   columns=['date', 'sales'])
                        historical_df['date'] = pd.to_datetime(historical_df['date'])
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=historical_df['date'],
                            y=historical_df['sales'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(forecast_df['date']),
                            y=forecast_df['forecast'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(title="Sales Forecast", xaxis_title="Date", yaxis_title="Sales")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info(f"Forecast generated using {data['method']} method")
                    else:
                        st.error("Failed to generate sales forecast")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Sales Analysis
        st.subheader("📊 Sales Analysis")
        
        sales_df = pd.DataFrame(self.sample_data.get('sales', []))
        if not sales_df.empty:
            sales_df['revenue'] = sales_df['amount'] * sales_df['quantity']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Revenue by Region")
                if 'region' in sales_df.columns:
                    region_revenue = sales_df.groupby('region')['revenue'].sum()
                    fig = px.bar(x=region_revenue.index, y=region_revenue.values, 
                               title="Revenue by Region")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top Products")
                if 'product_id' in sales_df.columns:
                    product_revenue = sales_df.groupby('product_id')['revenue'].sum().head(10)
                    fig = px.bar(x=product_revenue.index, y=product_revenue.values, 
                               title="Top 10 Products by Revenue")
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_pricing_optimization(self):
        """Render pricing optimization page"""
        st.header("💰 Pricing Optimization")
        
        if not self.sample_data:
            st.warning("Please load sample data first.")
            return
        
        # Price Optimization
        st.subheader("🎯 Price Optimization")
        
        if st.button("Optimize Prices", key="pricing_btn"):
            with st.spinner("Optimizing prices..."):
                try:
                    sales_data = self.sample_data.get('sales', [])[:100]
                    # Convert to proper format
                    price_data = []
                    for sale in sales_data:
                        price_data.append({
                            'product_id': sale.get('product_id', 'PROD_001'),
                            'price': sale.get('amount', 100),
                            'quantity': sale.get('quantity', 1)
                        })
                    
                    response = requests.post(
                        f"{self.api_base_url}/api/insights/price-optimization",
                        json={"price_data": price_data}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display optimization results
                        optimization_df = pd.DataFrame([
                            {
                                'product_id': product,
                                'current_price': result['current_price'],
                                'optimal_price': result['optimal_price'],
                                'price_change_percent': result['price_change_percent'],
                                'recommendation': result['recommendation']
                            }
                            for product, result in data['optimization_results'].items()
                        ])
                        
                        st.subheader("Price Optimization Results")
                        st.dataframe(optimization_df)
                        
                        # Summary
                        summary = data['summary']
                        st.markdown(f"""
                        <div class="insight-card">
                            <h4>Pricing Summary</h4>
                            <p><strong>Total Products:</strong> {summary['total_products']}</p>
                            <p><strong>Products to Increase:</strong> {summary['products_to_increase']}</p>
                            <p><strong>Products to Decrease:</strong> {summary['products_to_decrease']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Price change visualization
                        fig = px.bar(optimization_df, x='product_id', y='price_change_percent',
                                   title="Recommended Price Changes (%)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to optimize prices")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    def render_recommendations(self):
        """Render recommendations page"""
        st.header("💡 Business Recommendations")
        
        if not self.sample_data:
            st.warning("Please load sample data first.")
            return
        
        # Marketing Recommendations
        st.subheader("📢 Marketing Recommendations")
        
        if st.button("Generate Marketing Recommendations", key="marketing_rec_btn"):
            with st.spinner("Generating marketing recommendations..."):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/api/recommendations/marketing",
                        json={
                            "customer_data": self.sample_data.get('customers', [])[:100],
                            "n_clusters": 5
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display marketing recommendations
                        for segment, recommendations in data['marketing_recommendations'].items():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{segment.replace('_', ' ').title()}</h4>
                                <ul>
                                    {''.join([f'<li>{rec}</li>' for rec in recommendations[:5]])}
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to generate marketing recommendations")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Comprehensive Recommendations
        st.subheader("🎯 Comprehensive Business Recommendations")
        
        if st.button("Generate Comprehensive Recommendations", key="comprehensive_rec_btn"):
            with st.spinner("Generating comprehensive recommendations..."):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/api/recommendations/comprehensive",
                        json={
                            "customer_data": self.sample_data.get('customers', [])[:50],
                            "sales_data": self.sample_data.get('sales', [])[:100]
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display summary recommendations
                        if 'summary' in data and data['summary']:
                            st.markdown("### 📋 Executive Summary")
                            for recommendation in data['summary']:
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <p>{recommendation}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Display detailed recommendations
                        if 'marketing' in data and data['marketing']:
                            st.markdown("### 📢 Marketing Recommendations")
                            marketing_data = data['marketing']
                            if 'segment_recommendations' in marketing_data:
                                for segment, recommendations in marketing_data['segment_recommendations'].items():
                                    st.markdown(f"**{segment.replace('_', ' ').title()}:**")
                                    for rec in recommendations[:3]:
                                        st.markdown(f"- {rec}")
                        
                        if 'pricing' in data and data['pricing']:
                            st.markdown("### 💰 Pricing Recommendations")
                            pricing_data = data['pricing']
                            if 'price_optimization' in pricing_data:
                                st.info("Price optimization recommendations generated. Check the Pricing Optimization page for details.")
                    else:
                        st.error("Failed to generate comprehensive recommendations")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    def render_data_management(self):
        """Render data management page"""
        st.header("📁 Data Management")
        
        # Load Sample Data
        st.subheader("📊 Load Sample Data")
        
        if st.button("Load Sample Data", key="load_data_btn"):
            with st.spinner("Loading sample data..."):
                if self.load_sample_data():
                    st.success("Sample data loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load sample data")
        
        # Data Preview
        if self.sample_data:
            st.subheader("📋 Data Preview")
            
            tab1, tab2, tab3 = st.tabs(["Customers", "Sales", "Feedback"])
            
            with tab1:
                customers_df = pd.DataFrame(self.sample_data.get('customers', []))
                if not customers_df.empty:
                    st.dataframe(customers_df.head(10))
                    st.info(f"Total customers: {len(customers_df)}")
                else:
                    st.warning("No customer data available")
            
            with tab2:
                sales_df = pd.DataFrame(self.sample_data.get('sales', []))
                if not sales_df.empty:
                    st.dataframe(sales_df.head(10))
                    st.info(f"Total sales records: {len(sales_df)}")
                else:
                    st.warning("No sales data available")
            
            with tab3:
                feedback_df = pd.DataFrame(self.sample_data.get('feedback', []))
                if not feedback_df.empty:
                    st.dataframe(feedback_df.head(10))
                    st.info(f"Total feedback records: {len(feedback_df)}")
                else:
                    st.warning("No feedback data available")
    
    def render_settings(self):
        """Render settings page"""
        st.header("⚙️ Settings")
        
        st.subheader("🔗 API Configuration")
        
        api_url = st.text_input("API Base URL", value=self.api_base_url)
        if st.button("Update API URL"):
            self.api_base_url = api_url
            st.success("API URL updated!")
        
        st.subheader("📊 Dashboard Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Default Chart Type", ["Line", "Bar", "Pie"])
            st.selectbox("Color Theme", ["Blue", "Green", "Red", "Purple"])
        
        with col2:
            st.slider("Chart Height", 300, 800, 400)
            st.checkbox("Show Data Labels", value=True)
        
        st.subheader("🔔 Notifications")
        
        st.checkbox("Email notifications", value=False)
        st.checkbox("Browser notifications", value=True)
        st.text_input("Email address", placeholder="your@email.com")
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        # Load sample data if not already loaded
        if not self.sample_data:
            self.load_sample_data()
        
        # Render sidebar and get selected page
        page = self.render_sidebar()
        
        # Render selected page
        if page == "Overview":
            self.render_overview()
        elif page == "Customer Insights":
            self.render_customer_insights()
        elif page == "Sales Analytics":
            self.render_sales_analytics()
        elif page == "Pricing Optimization":
            self.render_pricing_optimization()
        elif page == "Recommendations":
            self.render_recommendations()
        elif page == "Data Management":
            self.render_data_management()
        elif page == "Settings":
            self.render_settings()

if __name__ == "__main__":
    dashboard = BusinessInsightsDashboard()
    dashboard.run()
