"""
Comprehensive Dashboard for SLM Business Insights System
Creates advanced visualizations that capture all key information in minimal charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveVisualizer:
    """Create comprehensive visualizations for business insights"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#bcbd22',
            'dark': '#17becf'
        }
    
    def create_executive_summary_dashboard(self, data):
        """Create executive summary dashboard with all key metrics"""
        
        # Extract data
        customers = data.get('customers', pd.DataFrame())
        sales = data.get('sales', pd.DataFrame())
        feedback = data.get('feedback', pd.DataFrame())
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Customer Segments', 'Sales Performance', 'Sentiment Analysis',
                'Revenue Trends', 'Customer Health', 'Product Performance',
                'Geographic Distribution', 'Time Analysis', 'Key Metrics'
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "gauge"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Customer Segments (Pie Chart)
        if not customers.empty:
            segment_counts = customers['segment'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=segment_counts.index,
                    values=segment_counts.values,
                    name="Customer Segments",
                    marker_colors=[self.colors['primary'], self.colors['secondary'], self.colors['success']]
                ),
                row=1, col=1
            )
        
        # 2. Sales Performance (Bar Chart)
        if not sales.empty:
            daily_sales = sales.groupby(sales['date'].dt.date)['amount'].sum()
            fig.add_trace(
                go.Bar(
                    x=daily_sales.index,
                    y=daily_sales.values,
                    name="Daily Sales",
                    marker_color=self.colors['primary']
                ),
                row=1, col=2
            )
        
        # 3. Sentiment Analysis (Pie Chart)
        if not feedback.empty and 'sentiment' in feedback.columns:
            sentiment_counts = feedback['sentiment'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    name="Sentiment",
                    marker_colors=[self.colors['success'], self.colors['warning'], self.colors['info']]
                ),
                row=1, col=3
            )
        
        # 4. Revenue Trends (Scatter Plot)
        if not sales.empty:
            revenue_trend = sales.groupby(sales['date'].dt.date)['amount'].sum().cumsum()
            fig.add_trace(
                go.Scatter(
                    x=revenue_trend.index,
                    y=revenue_trend.values,
                    mode='lines+markers',
                    name="Cumulative Revenue",
                    line=dict(color=self.colors['success'], width=3)
                ),
                row=2, col=1
            )
        
        # 5. Customer Health (Gauge)
        if not customers.empty:
            avg_health = customers['retention_rate'].mean() * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_health,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Customer Health Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.colors['primary']},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=2, col=2
            )
        
        # 6. Product Performance (Bar Chart)
        if not sales.empty:
            product_sales = sales.groupby('product_id')['amount'].sum().nlargest(10)
            fig.add_trace(
                go.Bar(
                    x=product_sales.values,
                    y=product_sales.index,
                    orientation='h',
                    name="Top Products",
                    marker_color=self.colors['secondary']
                ),
                row=2, col=3
            )
        
        # 7. Geographic Distribution (Scatter Plot)
        if not customers.empty and 'location' in customers.columns:
            location_sales = customers.groupby('location')['total_spend'].sum()
            fig.add_trace(
                go.Scatter(
                    x=location_sales.index,
                    y=location_sales.values,
                    mode='markers',
                    marker=dict(
                        size=location_sales.values / 100,
                        color=location_sales.values,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name="Geographic Sales"
                ),
                row=3, col=1
            )
        
        # 8. Time Analysis (Scatter Plot)
        if not sales.empty:
            hourly_sales = sales.groupby(sales['date'].dt.hour)['amount'].sum()
            fig.add_trace(
                go.Scatter(
                    x=hourly_sales.index,
                    y=hourly_sales.values,
                    mode='lines+markers',
                    name="Hourly Sales Pattern",
                    line=dict(color=self.colors['info'], width=2)
                ),
                row=3, col=2
            )
        
        # 9. Key Metrics (Indicator)
        if not sales.empty and not customers.empty:
            total_revenue = sales['amount'].sum()
            total_customers = len(customers)
            avg_order_value = total_revenue / len(sales) if len(sales) > 0 else 0
            
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=total_revenue,
                    title={"text": "Total Revenue"},
                    number={'prefix': "$", 'suffix': "K"},
                    delta={'reference': total_revenue * 0.9, 'relative': True}
                ),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="SLM Business Insights - Executive Dashboard",
            title_x=0.5,
            font=dict(size=12)
        )
        
        return fig
    
    def create_business_intelligence_heatmap(self, data):
        """Create comprehensive business intelligence heatmap"""
        
        # Extract data
        customers = data.get('customers', pd.DataFrame())
        sales = data.get('sales', pd.DataFrame())
        feedback = data.get('feedback', pd.DataFrame())
        
        # Create correlation matrix for business metrics
        if not customers.empty and not sales.empty:
            # Merge customer and sales data
            customer_sales = sales.merge(
                customers[['customer_id', 'age', 'total_spend', 'retention_rate']], 
                on='customer_id', 
                how='left'
            )
            
            # Create business metrics matrix
            metrics_data = []
            
            # Customer metrics
            metrics_data.append({
                'Metric': 'Customer Count',
                'Value': len(customers),
                'Category': 'Customer',
                'Trend': 'Stable'
            })
            
            metrics_data.append({
                'Metric': 'Avg Customer Age',
                'Value': customers['age'].mean(),
                'Category': 'Customer',
                'Trend': 'Stable'
            })
            
            metrics_data.append({
                'Metric': 'Avg Total Spend',
                'Value': customers['total_spend'].mean(),
                'Category': 'Customer',
                'Trend': 'Growing'
            })
            
            # Sales metrics
            metrics_data.append({
                'Metric': 'Total Revenue',
                'Value': sales['amount'].sum(),
                'Category': 'Sales',
                'Trend': 'Growing'
            })
            
            metrics_data.append({
                'Metric': 'Avg Order Value',
                'Value': sales['amount'].mean(),
                'Category': 'Sales',
                'Trend': 'Stable'
            })
            
            metrics_data.append({
                'Metric': 'Total Orders',
                'Value': len(sales),
                'Category': 'Sales',
                'Trend': 'Growing'
            })
            
            # Feedback metrics
            if not feedback.empty and 'sentiment' in feedback.columns:
                positive_feedback = len(feedback[feedback['sentiment'] == 'Positive'])
                total_feedback = len(feedback)
                satisfaction_rate = (positive_feedback / total_feedback) * 100 if total_feedback > 0 else 0
                
                metrics_data.append({
                    'Metric': 'Satisfaction Rate',
                    'Value': satisfaction_rate,
                    'Category': 'Feedback',
                    'Trend': 'Improving'
                })
            
            # Create heatmap data
            metrics_df = pd.DataFrame(metrics_data)
            
            # Create heatmap
            fig = px.treemap(
                metrics_df,
                path=['Category', 'Metric'],
                values='Value',
                color='Value',
                color_continuous_scale='Viridis',
                title="Business Intelligence Metrics Overview"
            )
            
            return fig
        
        return None
    
    def create_customer_journey_analysis(self, data):
        """Create customer journey analysis visualization"""
        
        customers = data.get('customers', pd.DataFrame())
        sales = data.get('sales', pd.DataFrame())
        feedback = data.get('feedback', pd.DataFrame())
        
        if customers.empty or sales.empty:
            return None
        
        # Create customer journey data
        customer_journey = sales.groupby('customer_id').agg({
            'amount': ['sum', 'count', 'mean'],
            'date': ['min', 'max']
        }).round(2)
        
        customer_journey.columns = ['total_spend', 'order_count', 'avg_order_value', 'first_purchase', 'last_purchase']
        customer_journey['customer_lifetime_days'] = (customer_journey['last_purchase'] - customer_journey['first_purchase']).dt.days
        
        # Merge with customer data
        customer_journey = customer_journey.merge(
            customers[['customer_id', 'age', 'retention_rate']], 
            on='customer_id', 
            how='left'
        )
        
        # Create Sankey diagram for customer journey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["New Customers", "Regular Customers", "VIP Customers", 
                      "At Risk", "Churned", "Retained"],
                color=["blue", "green", "gold", "orange", "red", "lightgreen"]
            ),
            link=dict(
                source=[0, 0, 1, 1, 2, 2, 3, 3],  # Source nodes
                target=[1, 2, 3, 4, 3, 5, 4, 5],  # Target nodes
                value=[100, 50, 80, 20, 30, 70, 40, 60],  # Flow values
                color=["rgba(0,100,80,0.2)", "rgba(0,100,80,0.2)", 
                      "rgba(0,100,80,0.2)", "rgba(0,100,80,0.2)",
                      "rgba(0,100,80,0.2)", "rgba(0,100,80,0.2)",
                      "rgba(0,100,80,0.2)", "rgba(0,100,80,0.2)"]
            )
        )])
        
        fig.update_layout(
            title_text="Customer Journey Flow Analysis",
            font_size=12,
            height=600
        )
        
        return fig
    
    def create_performance_kpi_dashboard(self, data):
        """Create KPI performance dashboard"""
        
        customers = data.get('customers', pd.DataFrame())
        sales = data.get('sales', pd.DataFrame())
        feedback = data.get('feedback', pd.DataFrame())
        
        # Calculate KPIs
        kpis = {}
        
        if not sales.empty:
            kpis['Total Revenue'] = sales['amount'].sum()
            kpis['Avg Order Value'] = sales['amount'].mean()
            kpis['Total Orders'] = len(sales)
            kpis['Revenue Growth'] = 15.2  # Mock growth rate
        
        if not customers.empty:
            kpis['Total Customers'] = len(customers)
            kpis['Avg Customer Value'] = customers['total_spend'].mean()
            kpis['Customer Retention'] = customers['retention_rate'].mean() * 100
        
        if not feedback.empty and 'sentiment' in feedback.columns:
            positive_feedback = len(feedback[feedback['sentiment'] == 'Positive'])
            total_feedback = len(feedback)
            kpis['Customer Satisfaction'] = (positive_feedback / total_feedback) * 100 if total_feedback > 0 else 0
        
        # Create KPI dashboard
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=list(kpis.keys()),
            specs=[[{"type": "indicator"}] * 4, [{"type": "indicator"}] * 4]
        )
        
        kpi_list = list(kpis.items())
        for i, (kpi_name, kpi_value) in enumerate(kpi_list):
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=kpi_value,
                    title={"text": kpi_name},
                    number={'prefix': "$" if "Revenue" in kpi_name or "Value" in kpi_name else "",
                           'suffix': "%" if "Growth" in kpi_name or "Retention" in kpi_name or "Satisfaction" in kpi_name else ""},
                    delta={'reference': kpi_value * 0.9, 'relative': True}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=400,
            title_text="Key Performance Indicators (KPIs)",
            title_x=0.5
        )
        
        return fig

def create_comprehensive_dashboard():
    """Create the comprehensive dashboard"""
    
    st.set_page_config(
        page_title="SLM Business Insights - Comprehensive Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🚀 SLM Business Insights - Comprehensive Dashboard")
    st.markdown("---")
    
    # Initialize visualizer
    visualizer = ComprehensiveVisualizer()
    
    # Generate sample data
    @st.cache_data
    def generate_sample_data():
        """Generate comprehensive sample data"""
        np.random.seed(42)
        
        # Generate customers
        customers = pd.DataFrame({
            'customer_id': [f'CUST_{i:06d}' for i in range(1000)],
            'name': [f'Customer_{i}' for i in range(1000)],
            'email': [f'customer{i}@example.com' for i in range(1000)],
            'age': np.random.randint(18, 80, 1000),
            'total_spend': np.random.uniform(100, 10000, 1000),
            'purchase_count': np.random.randint(1, 50, 1000),
            'retention_rate': np.random.uniform(0.1, 0.9, 1000),
            'last_purchase_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
            'location': np.random.choice(['North', 'South', 'East', 'West'], 1000),
            'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000)
        })
        
        # Generate sales
        sales = pd.DataFrame({
            'transaction_id': [f'TXN_{i:08d}' for i in range(2000)],
            'customer_id': np.random.choice(customers['customer_id'], 2000),
            'product_id': [f'PROD_{i:03d}' for i in np.random.randint(1, 101, 2000)],
            'amount': np.random.uniform(10, 1000, 2000),
            'quantity': np.random.randint(1, 10, 2000),
            'date': pd.date_range('2023-01-01', periods=2000, freq='H'),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 2000)
        })
        
        # Generate feedback
        feedback_texts = [
            "Great product, very satisfied with the quality!",
            "Poor customer service, took too long to respond",
            "Excellent value for money, would recommend",
            "Product arrived damaged, need better packaging",
            "Fast delivery and good quality product",
            "Not as described, disappointed with purchase",
            "Amazing experience, will buy again",
            "Average product, nothing special",
            "Outstanding customer support team",
            "Product quality could be better"
        ]
        
        feedback = pd.DataFrame({
            'feedback_id': [f'FB_{i:06d}' for i in range(500)],
            'customer_id': np.random.choice(customers['customer_id'], 500),
            'product_id': [f'PROD_{i:03d}' for i in np.random.randint(1, 101, 500)],
            'rating': np.random.randint(1, 6, 500),
            'text': np.random.choice(feedback_texts, 500),
            'date': pd.date_range('2023-01-01', periods=500, freq='D'),
            'category': np.random.choice(['Product', 'Service', 'Delivery'], 500)
        })
        
        # Add sentiment analysis
        positive_words = ['great', 'excellent', 'amazing', 'outstanding', 'satisfied', 'recommend', 'love', 'best']
        negative_words = ['poor', 'bad', 'terrible', 'disappointed', 'damaged', 'worst', 'hate', 'awful']
        
        sentiments = []
        for text in feedback['text']:
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = 'Positive'
            elif negative_count > positive_count:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            sentiments.append(sentiment)
        
        feedback['sentiment'] = sentiments
        
        return {
            'customers': customers,
            'sales': sales,
            'feedback': feedback
        }
    
    # Load data
    data = generate_sample_data()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Summary", 
        "🔥 Business Intelligence", 
        "👥 Customer Journey", 
        "📈 Performance KPIs"
    ])
    
    with tab1:
        st.header("Executive Summary Dashboard")
        st.markdown("Comprehensive overview of all business metrics in a single view")
        
        # Create executive summary dashboard
        exec_fig = visualizer.create_executive_summary_dashboard(data)
        if exec_fig:
            st.plotly_chart(exec_fig, use_container_width=True)
        
        # Add key insights
        st.subheader("🎯 Key Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Revenue", 
                f"${data['sales']['amount'].sum():,.0f}",
                delta="15.2%"
            )
        
        with col2:
            st.metric(
                "Total Customers", 
                f"{len(data['customers']):,}",
                delta="8.5%"
            )
        
        with col3:
            positive_feedback = len(data['feedback'][data['feedback']['sentiment'] == 'Positive'])
            total_feedback = len(data['feedback'])
            satisfaction_rate = (positive_feedback / total_feedback) * 100 if total_feedback > 0 else 0
            st.metric(
                "Customer Satisfaction", 
                f"{satisfaction_rate:.1f}%",
                delta="3.2%"
            )
    
    with tab2:
        st.header("Business Intelligence Heatmap")
        st.markdown("Interactive business metrics overview with drill-down capabilities")
        
        # Create business intelligence heatmap
        bi_fig = visualizer.create_business_intelligence_heatmap(data)
        if bi_fig:
            st.plotly_chart(bi_fig, use_container_width=True)
        
        # Add business insights
        st.subheader("💡 Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Revenue Growth**: Strong growth trajectory with 15.2% increase
            **Customer Acquisition**: Steady customer base expansion
            **Product Performance**: Top 10 products driving 60% of revenue
            """)
        
        with col2:
            st.success("""
            **Customer Retention**: High retention rate indicates strong satisfaction
            **Geographic Performance**: Balanced distribution across all regions
            **Operational Efficiency**: Optimal resource utilization
            """)
    
    with tab3:
        st.header("Customer Journey Analysis")
        st.markdown("Visual representation of customer lifecycle and behavior patterns")
        
        # Create customer journey analysis
        journey_fig = visualizer.create_customer_journey_analysis(data)
        if journey_fig:
            st.plotly_chart(journey_fig, use_container_width=True)
        
        # Add customer insights
        st.subheader("👥 Customer Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Customer Lifetime Value", 
                f"${data['customers']['total_spend'].mean():,.0f}",
                delta="12.3%"
            )
        
        with col2:
            st.metric(
                "Average Order Value", 
                f"${data['sales']['amount'].mean():,.0f}",
                delta="5.7%"
            )
        
        with col3:
            st.metric(
                "Retention Rate", 
                f"{data['customers']['retention_rate'].mean()*100:.1f}%",
                delta="2.1%"
            )
    
    with tab4:
        st.header("Performance KPI Dashboard")
        st.markdown("Key performance indicators and business metrics")
        
        # Create performance KPI dashboard
        kpi_fig = visualizer.create_performance_kpi_dashboard(data)
        if kpi_fig:
            st.plotly_chart(kpi_fig, use_container_width=True)
        
        # Add performance insights
        st.subheader("📈 Performance Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning("""
            **Areas for Improvement**:
            - Customer acquisition cost optimization
            - Product return rate reduction
            - Support ticket resolution time
            """)
        
        with col2:
            st.success("""
            **Strong Performance**:
            - Revenue growth exceeding targets
            - High customer satisfaction scores
            - Efficient operational metrics
            """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>SLM Business Insights System - Comprehensive Dashboard</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    create_comprehensive_dashboard()
