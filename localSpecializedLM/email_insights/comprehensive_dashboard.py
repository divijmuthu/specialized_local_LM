"""
Comprehensive Dashboard for Email Insights Application
Creates advanced visualizations that capture all email analysis information in minimal charts
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

class EmailInsightsVisualizer:
    """Create comprehensive visualizations for email insights"""
    
    def __init__(self):
        self.colors = {
            'positive': '#2ca02c',
            'negative': '#d62728',
            'neutral': '#ff7f0e',
            'complaint': '#d62728',
            'inquiry': '#1f77b4',
            'feedback': '#9467bd',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e'
        }
    
    def create_email_analysis_dashboard(self, email_data):
        """Create comprehensive email analysis dashboard"""
        
        if email_data.empty:
            return None
        
        # Create subplots for comprehensive view
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Email Volume Over Time', 'Sentiment Distribution', 'Email Type Analysis',
                'Sender Analysis', 'Response Time Patterns', 'Topic Clusters',
                'Email Length Analysis', 'Priority Classification', 'Key Metrics'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "pie"}],
                [{"type": "histogram"}, {"type": "bar"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Email Volume Over Time
        if 'date' in email_data.columns:
            daily_volume = email_data.groupby(email_data['date'].dt.date).size()
            fig.add_trace(
                go.Scatter(
                    x=daily_volume.index,
                    y=daily_volume.values,
                    mode='lines+markers',
                    name="Daily Email Volume",
                    line=dict(color=self.colors['primary'], width=3)
                ),
                row=1, col=1
            )
        
        # 2. Sentiment Distribution
        if 'sentiment' in email_data.columns:
            sentiment_counts = email_data['sentiment'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    name="Sentiment",
                    marker_colors=[self.colors['positive'], self.colors['negative'], self.colors['neutral']]
                ),
                row=1, col=2
            )
        
        # 3. Email Type Analysis
        if 'email_type' in email_data.columns:
            type_counts = email_data['email_type'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    name="Email Types",
                    marker_color=[self.colors['complaint'], self.colors['inquiry'], self.colors['feedback']]
                ),
                row=1, col=3
            )
        
        # 4. Sender Analysis
        if 'sender' in email_data.columns:
            top_senders = email_data['sender'].value_counts().head(10)
            fig.add_trace(
                go.Bar(
                    x=top_senders.values,
                    y=top_senders.index,
                    orientation='h',
                    name="Top Senders",
                    marker_color=self.colors['secondary']
                ),
                row=2, col=1
            )
        
        # 5. Response Time Patterns
        if 'date' in email_data.columns:
            hourly_volume = email_data.groupby(email_data['date'].dt.hour).size()
            fig.add_trace(
                go.Scatter(
                    x=hourly_volume.index,
                    y=hourly_volume.values,
                    mode='lines+markers',
                    name="Hourly Email Volume",
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=2, col=2
            )
        
        # 6. Topic Clusters
        if 'topic_cluster' in email_data.columns:
            topic_counts = email_data['topic_cluster'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=[f"Topic {i}" for i in topic_counts.index],
                    values=topic_counts.values,
                    name="Topic Clusters",
                    marker_colors=px.colors.qualitative.Set3
                ),
                row=2, col=3
            )
        
        # 7. Email Length Analysis
        if 'body' in email_data.columns:
            email_lengths = email_data['body'].str.len()
            fig.add_trace(
                go.Histogram(
                    x=email_lengths,
                    name="Email Length Distribution",
                    marker_color=self.colors['primary'],
                    nbinsx=20
                ),
                row=3, col=1
            )
        
        # 8. Priority Classification
        if 'sentiment' in email_data.columns and 'email_type' in email_data.columns:
            # Create priority based on sentiment and type
            priority_data = email_data.copy()
            priority_data['priority'] = priority_data.apply(
                lambda row: 'High' if (row['sentiment'] == 'Negative' and row['email_type'] == 'complaint') 
                           else 'Medium' if row['sentiment'] == 'Negative' or row['email_type'] == 'complaint'
                           else 'Low', axis=1
            )
            
            priority_counts = priority_data['priority'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=priority_counts.index,
                    y=priority_counts.values,
                    name="Priority Classification",
                    marker_color=[self.colors['negative'], self.colors['secondary'], self.colors['positive']]
                ),
                row=3, col=2
            )
        
        # 9. Key Metrics
        total_emails = len(email_data)
        if 'sentiment' in email_data.columns:
            positive_emails = len(email_data[email_data['sentiment'] == 'Positive'])
            satisfaction_rate = (positive_emails / total_emails) * 100 if total_emails > 0 else 0
        else:
            satisfaction_rate = 0
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=satisfaction_rate,
                title={"text": "Customer Satisfaction Rate"},
                number={'suffix': "%"},
                delta={'reference': satisfaction_rate * 0.9, 'relative': True}
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="Email Insights - Comprehensive Analysis Dashboard",
            title_x=0.5,
            font=dict(size=12)
        )
        
        return fig
    
    def create_sentiment_trend_analysis(self, email_data):
        """Create sentiment trend analysis over time"""
        
        if email_data.empty or 'sentiment' not in email_data.columns or 'date' not in email_data.columns:
            return None
        
        # Create sentiment trends over time
        sentiment_trends = email_data.groupby([email_data['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        for sentiment in sentiment_trends.columns:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_trends.index,
                    y=sentiment_trends[sentiment],
                    mode='lines+markers',
                    name=sentiment,
                    line=dict(width=3),
                    marker=dict(size=6)
                )
            )
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Emails",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_email_flow_analysis(self, email_data):
        """Create email flow and response analysis"""
        
        if email_data.empty:
            return None
        
        # Create email flow data
        flow_data = []
        
        if 'email_type' in email_data.columns:
            type_counts = email_data['email_type'].value_counts()
            for email_type, count in type_counts.items():
                flow_data.append({
                    'Category': 'Email Type',
                    'Value': email_type,
                    'Count': count,
                    'Percentage': (count / len(email_data)) * 100
                })
        
        if 'sentiment' in email_data.columns:
            sentiment_counts = email_data['sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                flow_data.append({
                    'Category': 'Sentiment',
                    'Value': sentiment,
                    'Count': count,
                    'Percentage': (count / len(email_data)) * 100
                })
        
        if flow_data:
            flow_df = pd.DataFrame(flow_data)
            
            fig = px.treemap(
                flow_df,
                path=['Category', 'Value'],
                values='Count',
                color='Percentage',
                color_continuous_scale='Viridis',
                title="Email Analysis Flow - Type and Sentiment Distribution"
            )
            
            return fig
        
        return None
    
    def create_performance_metrics(self, email_data):
        """Create performance metrics dashboard"""
        
        if email_data.empty:
            return None
        
        # Calculate metrics
        metrics = {}
        
        metrics['Total Emails'] = len(email_data)
        
        if 'sentiment' in email_data.columns:
            positive_emails = len(email_data[email_data['sentiment'] == 'Positive'])
            metrics['Positive Emails'] = positive_emails
            metrics['Satisfaction Rate'] = (positive_emails / len(email_data)) * 100
        
        if 'email_type' in email_data.columns:
            complaint_emails = len(email_data[email_data['email_type'] == 'complaint'])
            metrics['Complaint Rate'] = (complaint_emails / len(email_data)) * 100
        
        if 'date' in email_data.columns:
            # Calculate response time metrics
            email_data['date'] = pd.to_datetime(email_data['date'])
            time_span = (email_data['date'].max() - email_data['date'].min()).days
            metrics['Avg Emails per Day'] = len(email_data) / time_span if time_span > 0 else 0
        
        # Create metrics dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=list(metrics.keys()),
            specs=[[{"type": "indicator"}] * 3, [{"type": "indicator"}] * 3]
        )
        
        metric_list = list(metrics.items())
        for i, (metric_name, metric_value) in enumerate(metric_list):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=metric_value,
                    title={"text": metric_name},
                    number={'suffix': "%" if "Rate" in metric_name else ""}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=400,
            title_text="Email Analysis Performance Metrics",
            title_x=0.5
        )
        
        return fig

def create_comprehensive_email_dashboard():
    """Create the comprehensive email insights dashboard"""
    
    st.set_page_config(
        page_title="Email Insights - Comprehensive Dashboard",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 Email Insights - Comprehensive Dashboard")
    st.markdown("---")
    
    # Initialize visualizer
    visualizer = EmailInsightsVisualizer()
    
    # Generate sample email data
    @st.cache_data
    def generate_sample_email_data():
        """Generate comprehensive sample email data"""
        np.random.seed(42)
        
        # Generate email data
        n_emails = 500
        
        # Email subjects
        subjects = [
            "Product inquiry", "Complaint about service", "Thank you for your help",
            "Order status question", "Refund request", "Product feedback",
            "Technical support needed", "Billing inquiry", "Feature request",
            "Account issue", "Shipping question", "Product recommendation"
        ]
        
        # Email senders
        senders = [
            "customer1@example.com", "customer2@example.com", "customer3@example.com",
            "customer4@example.com", "customer5@example.com", "customer6@example.com",
            "customer7@example.com", "customer8@example.com", "customer9@example.com",
            "customer10@example.com"
        ]
        
        # Email bodies
        bodies = [
            "I am very satisfied with your product. It works perfectly and exceeded my expectations.",
            "I am disappointed with the service I received. The product arrived damaged and customer service was unhelpful.",
            "Can you help me with my order? I have a question about the shipping status.",
            "Thank you for your excellent customer service. You resolved my issue quickly and professionally.",
            "I need a refund for my recent purchase. The product doesn't work as advertised.",
            "I love this product! It's exactly what I was looking for and the quality is outstanding.",
            "I'm having trouble with my account. Can you please help me reset my password?",
            "The product is okay, but I think it could be improved in a few areas.",
            "I would like to request a new feature for your product. It would be very helpful.",
            "I'm not happy with the billing. There seems to be an error in my invoice."
        ]
        
        # Generate email data
        email_data = pd.DataFrame({
            'email_id': [f'EMAIL_{i:06d}' for i in range(n_emails)],
            'subject': np.random.choice(subjects, n_emails),
            'sender': np.random.choice(senders, n_emails),
            'body': np.random.choice(bodies, n_emails),
            'date': pd.date_range('2023-01-01', periods=n_emails, freq='H'),
            'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], n_emails, p=[0.4, 0.3, 0.3]),
            'email_type': np.random.choice(['complaint', 'inquiry', 'feedback'], n_emails, p=[0.2, 0.5, 0.3]),
            'topic_cluster': np.random.randint(0, 5, n_emails)
        })
        
        return email_data
    
    # Load data
    email_data = generate_sample_email_data()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Email Analysis Overview", 
        "📈 Sentiment Trends", 
        "🔄 Email Flow Analysis", 
        "📋 Performance Metrics"
    ])
    
    with tab1:
        st.header("Email Analysis Overview")
        st.markdown("Comprehensive analysis of all email metrics in a single view")
        
        # Create email analysis dashboard
        analysis_fig = visualizer.create_email_analysis_dashboard(email_data)
        if analysis_fig:
            st.plotly_chart(analysis_fig, use_container_width=True)
        
        # Add key insights
        st.subheader("🎯 Key Email Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Emails", 
                f"{len(email_data):,}",
                delta="12.5%"
            )
        
        with col2:
            positive_emails = len(email_data[email_data['sentiment'] == 'Positive'])
            satisfaction_rate = (positive_emails / len(email_data)) * 100
            st.metric(
                "Satisfaction Rate", 
                f"{satisfaction_rate:.1f}%",
                delta="3.2%"
            )
        
        with col3:
            complaint_emails = len(email_data[email_data['email_type'] == 'complaint'])
            complaint_rate = (complaint_emails / len(email_data)) * 100
            st.metric(
                "Complaint Rate", 
                f"{complaint_rate:.1f}%",
                delta="-1.5%"
            )
    
    with tab2:
        st.header("Sentiment Trends Over Time")
        st.markdown("Track sentiment patterns and trends across time periods")
        
        # Create sentiment trend analysis
        trend_fig = visualizer.create_sentiment_trend_analysis(email_data)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
        
        # Add sentiment insights
        st.subheader("😊 Sentiment Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Positive Trends**:
            - Customer satisfaction improving over time
            - Positive feedback increasing
            - Service quality enhancements showing results
            """)
        
        with col2:
            st.warning("""
            **Areas for Improvement**:
            - Negative sentiment spikes during peak periods
            - Complaint resolution time needs optimization
            - Proactive customer communication required
            """)
    
    with tab3:
        st.header("Email Flow Analysis")
        st.markdown("Visual representation of email types and sentiment distribution")
        
        # Create email flow analysis
        flow_fig = visualizer.create_email_flow_analysis(email_data)
        if flow_fig:
            st.plotly_chart(flow_fig, use_container_width=True)
        
        # Add flow insights
        st.subheader("🔄 Email Flow Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            inquiry_emails = len(email_data[email_data['email_type'] == 'inquiry'])
            st.metric(
                "Inquiry Emails", 
                f"{inquiry_emails:,}",
                delta="8.3%"
            )
        
        with col2:
            feedback_emails = len(email_data[email_data['email_type'] == 'feedback'])
            st.metric(
                "Feedback Emails", 
                f"{feedback_emails:,}",
                delta="15.7%"
            )
        
        with col3:
            neutral_emails = len(email_data[email_data['sentiment'] == 'Neutral'])
            st.metric(
                "Neutral Emails", 
                f"{neutral_emails:,}",
                delta="2.1%"
            )
    
    with tab4:
        st.header("Performance Metrics")
        st.markdown("Key performance indicators and email analysis metrics")
        
        # Create performance metrics
        metrics_fig = visualizer.create_performance_metrics(email_data)
        if metrics_fig:
            st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Add performance insights
        st.subheader("📈 Performance Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **Strong Performance**:
            - High email response rate
            - Good sentiment distribution
            - Efficient email processing
            """)
        
        with col2:
            st.info("""
            **Optimization Opportunities**:
            - Reduce complaint resolution time
            - Improve inquiry response quality
            - Enhance proactive communication
            """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Email Insights Application - Comprehensive Dashboard</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    create_comprehensive_email_dashboard()
