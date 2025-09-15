"""
Comprehensive Dashboard for Customer Service Feedback Analysis Tool
Creates advanced visualizations that capture all customer service insights in minimal charts
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

class CustomerServiceVisualizer:
    """Create comprehensive visualizations for customer service insights"""
    
    def __init__(self):
        self.colors = {
            'positive': '#2ca02c',
            'negative': '#d62728',
            'neutral': '#ff7f0e',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
    
    def create_customer_service_dashboard(self, feedback_data):
        """Create comprehensive customer service dashboard"""
        
        if feedback_data.empty:
            return None
        
        # Create subplots for comprehensive view
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Feedback Volume Over Time', 'Sentiment Distribution', 'Source Analysis',
                'Topic Distribution', 'Actionable Items', 'Response Time Analysis',
                'Customer Satisfaction Trends', 'Priority Classification', 'Key Metrics'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Feedback Volume Over Time
        if 'date' in feedback_data.columns:
            daily_volume = feedback_data.groupby(feedback_data['date'].dt.date).size()
            fig.add_trace(
                go.Scatter(
                    x=daily_volume.index,
                    y=daily_volume.values,
                    mode='lines+markers',
                    name="Daily Feedback Volume",
                    line=dict(color=self.colors['primary'], width=3)
                ),
                row=1, col=1
            )
        
        # 2. Sentiment Distribution
        if 'sentiment' in feedback_data.columns:
            sentiment_counts = feedback_data['sentiment'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    name="Sentiment",
                    marker_colors=[self.colors['positive'], self.colors['negative'], self.colors['neutral']]
                ),
                row=1, col=2
            )
        
        # 3. Source Analysis
        if 'source' in feedback_data.columns:
            source_counts = feedback_data['source'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=source_counts.index,
                    y=source_counts.values,
                    name="Feedback Sources",
                    marker_color=self.colors['secondary']
                ),
                row=1, col=3
            )
        
        # 4. Topic Distribution
        if 'topic_label' in feedback_data.columns:
            topic_counts = feedback_data['topic_label'].value_counts().head(5)
            fig.add_trace(
                go.Pie(
                    labels=topic_counts.index,
                    values=topic_counts.values,
                    name="Top Topics",
                    marker_colors=px.colors.qualitative.Set3
                ),
                row=2, col=1
            )
        
        # 5. Actionable Items
        if 'actionable_items' in feedback_data.columns:
            all_actions = []
            for actions in feedback_data['actionable_items']:
                if isinstance(actions, list):
                    all_actions.extend([action[0] for action in actions])
            
            if all_actions:
                action_counts = pd.Series(all_actions).value_counts().head(5)
                fig.add_trace(
                    go.Bar(
                        x=action_counts.values,
                        y=action_counts.index,
                        orientation='h',
                        name="Actionable Items",
                        marker_color=self.colors['warning']
                    ),
                    row=2, col=2
                )
        
        # 6. Response Time Analysis (simulated)
        if 'date' in feedback_data.columns:
            # Simulate response time based on sentiment
            response_times = []
            for sentiment in feedback_data['sentiment']:
                if sentiment == 'negative':
                    response_times.append(np.random.uniform(2, 8))  # Hours
                elif sentiment == 'positive':
                    response_times.append(np.random.uniform(0.5, 2))
                else:
                    response_times.append(np.random.uniform(1, 4))
            
            fig.add_trace(
                go.Scatter(
                    x=feedback_data['date'],
                    y=response_times,
                    mode='markers',
                    name="Response Time",
                    marker=dict(
                        color=response_times,
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Hours")
                    )
                ),
                row=2, col=3
            )
        
        # 7. Customer Satisfaction Trends
        if 'date' in feedback_data.columns and 'sentiment' in feedback_data.columns:
            monthly_satisfaction = feedback_data.groupby([
                feedback_data['date'].dt.to_period('M'), 'sentiment'
            ]).size().unstack(fill_value=0)
            
            if 'positive' in monthly_satisfaction.columns:
                fig.add_trace(
                    go.Scatter(
                        x=monthly_satisfaction.index.astype(str),
                        y=monthly_satisfaction['positive'],
                        mode='lines+markers',
                        name="Positive Feedback Trend",
                        line=dict(color=self.colors['positive'], width=3)
                    ),
                    row=3, col=1
                )
        
        # 8. Priority Classification
        if 'sentiment' in feedback_data.columns:
            # Create priority based on sentiment
            priority_data = feedback_data.copy()
            priority_data['priority'] = priority_data['sentiment'].apply(
                lambda x: 'High' if x == 'negative' else 'Medium' if x == 'neutral' else 'Low'
            )
            
            priority_counts = priority_data['priority'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=priority_counts.index,
                    y=priority_counts.values,
                    name="Priority Classification",
                    marker_color=[self.colors['warning'], self.colors['secondary'], self.colors['success']]
                ),
                row=3, col=2
            )
        
        # 9. Key Metrics
        if 'sentiment' in feedback_data.columns:
            positive_feedback = len(feedback_data[feedback_data['sentiment'] == 'positive'])
            total_feedback = len(feedback_data)
            satisfaction_rate = (positive_feedback / total_feedback) * 100 if total_feedback > 0 else 0
            
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
            title_text="Customer Service Feedback Analysis - Comprehensive Dashboard",
            title_x=0.5,
            font=dict(size=12)
        )
        
        return fig
    
    def create_sentiment_trend_analysis(self, feedback_data):
        """Create sentiment trend analysis over time"""
        
        if feedback_data.empty or 'sentiment' not in feedback_data.columns or 'date' not in feedback_data.columns:
            return None
        
        # Create sentiment trends over time
        sentiment_trends = feedback_data.groupby([feedback_data['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
        
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
            title="Customer Sentiment Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Feedback Items",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_actionable_insights_analysis(self, feedback_data):
        """Create actionable insights analysis"""
        
        if feedback_data.empty or 'actionable_items' not in feedback_data.columns:
            return None
        
        # Extract all actionable items
        all_actions = []
        for actions in feedback_data['actionable_items']:
            if isinstance(actions, list):
                all_actions.extend([action[0] for action in actions])
        
        if not all_actions:
            return None
        
        # Create actionable insights data
        action_data = []
        action_counts = pd.Series(all_actions).value_counts()
        
        for action, count in action_counts.items():
            action_data.append({
                'Category': 'Actionable Items',
                'Value': action,
                'Count': count,
                'Percentage': (count / len(all_actions)) * 100
            })
        
        if action_data:
            action_df = pd.DataFrame(action_data)
            
            fig = px.treemap(
                action_df,
                path=['Category', 'Value'],
                values='Count',
                color='Percentage',
                color_continuous_scale='Viridis',
                title="Actionable Insights - Priority Analysis"
            )
            
            return fig
        
        return None
    
    def create_customer_journey_analysis(self, feedback_data):
        """Create customer journey analysis"""
        
        if feedback_data.empty:
            return None
        
        # Create customer journey data
        journey_data = []
        
        if 'source' in feedback_data.columns:
            source_counts = feedback_data['source'].value_counts()
            for source, count in source_counts.items():
                journey_data.append({
                    'Category': 'Feedback Source',
                    'Value': source,
                    'Count': count,
                    'Percentage': (count / len(feedback_data)) * 100
                })
        
        if 'sentiment' in feedback_data.columns:
            sentiment_counts = feedback_data['sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                journey_data.append({
                    'Category': 'Sentiment',
                    'Value': sentiment,
                    'Count': count,
                    'Percentage': (count / len(feedback_data)) * 100
                })
        
        if journey_data:
            journey_df = pd.DataFrame(journey_data)
            
            fig = px.treemap(
                journey_df,
                path=['Category', 'Value'],
                values='Count',
                color='Percentage',
                color_continuous_scale='Blues',
                title="Customer Journey Analysis - Source and Sentiment Distribution"
            )
            
            return fig
        
        return None
    
    def create_performance_metrics(self, feedback_data):
        """Create performance metrics dashboard"""
        
        if feedback_data.empty:
            return None
        
        # Calculate metrics
        metrics = {}
        
        metrics['Total Feedback'] = len(feedback_data)
        
        if 'sentiment' in feedback_data.columns:
            positive_feedback = len(feedback_data[feedback_data['sentiment'] == 'positive'])
            negative_feedback = len(feedback_data[feedback_data['sentiment'] == 'negative'])
            metrics['Positive Feedback'] = positive_feedback
            metrics['Negative Feedback'] = negative_feedback
            metrics['Satisfaction Rate'] = (positive_feedback / len(feedback_data)) * 100
            metrics['Complaint Rate'] = (negative_feedback / len(feedback_data)) * 100
        
        if 'actionable_items' in feedback_data.columns:
            all_actions = []
            for actions in feedback_data['actionable_items']:
                if isinstance(actions, list):
                    all_actions.extend([action[0] for action in actions])
            metrics['Actionable Items'] = len(all_actions)
        
        if 'date' in feedback_data.columns:
            # Calculate response time metrics
            feedback_data['date'] = pd.to_datetime(feedback_data['date'])
            time_span = (feedback_data['date'].max() - feedback_data['date'].min()).days
            metrics['Avg Feedback per Day'] = len(feedback_data) / time_span if time_span > 0 else 0
        
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
            title_text="Customer Service Performance Metrics",
            title_x=0.5
        )
        
        return fig

def create_comprehensive_customer_service_dashboard():
    """Create the comprehensive customer service dashboard"""
    
    st.set_page_config(
        page_title="Customer Service Feedback Analysis - Comprehensive Dashboard",
        page_icon="🎧",
        layout="wide"
    )
    
    st.title("🎧 Customer Service Feedback Analysis - Comprehensive Dashboard")
    st.markdown("---")
    
    # Initialize visualizer
    visualizer = CustomerServiceVisualizer()
    
    # Generate sample feedback data
    @st.cache_data
    def generate_sample_feedback_data():
        """Generate comprehensive sample feedback data"""
        np.random.seed(42)
        
        # Generate feedback data
        n_feedback = 500
        
        # Feedback texts
        feedback_texts = [
            "I love your product! It's amazing and works perfectly.",
            "Customer service was terrible. They didn't respond to my issue.",
            "The product is good, but the pricing is too high.",
            "Excellent support team! They resolved my issue quickly.",
            "The new feature is great! It solves a major problem for us.",
            "I've been a customer for years, but recent changes are disappointing.",
            "Fast delivery and good quality product. Highly recommend!",
            "Poor quality product. Not worth the money.",
            "The support team was very helpful in resolving my issue.",
            "The product works well, but the documentation is lacking.",
            "There is a bug in the login system that needs to be fixed",
            "The documentation is poor and needs improvement",
            "Customer service response time is too slow",
            "The pricing is too high compared to competitors",
            "The user interface is confusing and needs redesign",
            "Please add a dark mode feature to the application",
            "The product crashes frequently and needs stability fixes",
            "The support team needs better training",
            "The mobile app is missing important features",
            "The checkout process is too complicated"
        ]
        
        # Feedback sources
        sources = ['email', 'review', 'survey', 'social_media', 'support_ticket']
        
        # Generate feedback data
        feedback_data = pd.DataFrame({
            'feedback_id': [f'FB_{i:06d}' for i in range(n_feedback)],
            'text': np.random.choice(feedback_texts, n_feedback),
            'date': pd.date_range('2023-01-01', periods=n_feedback, freq='H'),
            'source': np.random.choice(sources, n_feedback),
            'customer_id': [f'CUST_{i:06d}' for i in np.random.randint(1, 100, n_feedback)]
        })
        
        # Add sentiment analysis
        positive_words = ['love', 'excellent', 'amazing', 'great', 'fantastic', 'helpful', 'recommend', 'perfect']
        negative_words = ['terrible', 'poor', 'disappointing', 'hate', 'worst', 'bad', 'slow', 'confusing']
        
        sentiments = []
        for text in feedback_data['text']:
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = 'positive'
            elif negative_count > positive_count:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            sentiments.append(sentiment)
        
        feedback_data['sentiment'] = sentiments
        
        # Add topic labels
        topics = ['product_quality', 'customer_service', 'pricing', 'features', 'documentation', 'ui_ux', 'bugs', 'support']
        feedback_data['topic_label'] = np.random.choice(topics, n_feedback)
        
        # Add actionable items
        actionable_items = []
        for text in feedback_data['text']:
            actions = []
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['bug', 'error', 'crash', 'not working']):
                actions.append(('bug', 'bug_fix'))
            if any(word in text_lower for word in ['documentation', 'manual', 'hard to understand']):
                actions.append(('documentation', 'improve_docs'))
            if any(word in text_lower for word in ['slow', 'response time', 'didn\'t respond']):
                actions.append(('response time', 'improve_response'))
            if any(word in text_lower for word in ['pricing', 'price', 'expensive', 'cost']):
                actions.append(('pricing', 'review_pricing'))
            if any(word in text_lower for word in ['confusing', 'interface', 'design']):
                actions.append(('ui/ux', 'improve_ui'))
            if any(word in text_lower for word in ['add', 'feature', 'wish', 'would like']):
                actions.append(('feature request', 'new_feature'))
            
            actionable_items.append(actions)
        
        feedback_data['actionable_items'] = actionable_items
        
        return feedback_data
    
    # Load data
    feedback_data = generate_sample_feedback_data()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Feedback Analysis Overview", 
        "📈 Sentiment Trends", 
        "🎯 Actionable Insights", 
        "📋 Performance Metrics"
    ])
    
    with tab1:
        st.header("Feedback Analysis Overview")
        st.markdown("Comprehensive analysis of all customer service feedback metrics in a single view")
        
        # Create feedback analysis dashboard
        analysis_fig = visualizer.create_customer_service_dashboard(feedback_data)
        if analysis_fig:
            st.plotly_chart(analysis_fig, use_container_width=True)
        
        # Add key insights
        st.subheader("🎯 Key Customer Service Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Feedback", 
                f"{len(feedback_data):,}",
                delta="15.2%"
            )
        
        with col2:
            positive_feedback = len(feedback_data[feedback_data['sentiment'] == 'positive'])
            satisfaction_rate = (positive_feedback / len(feedback_data)) * 100
            st.metric(
                "Satisfaction Rate", 
                f"{satisfaction_rate:.1f}%",
                delta="3.2%"
            )
        
        with col3:
            all_actions = []
            for actions in feedback_data['actionable_items']:
                all_actions.extend([action[0] for action in actions])
            st.metric(
                "Actionable Items", 
                f"{len(all_actions):,}",
                delta="8.5%"
            )
    
    with tab2:
        st.header("Sentiment Trends Over Time")
        st.markdown("Track customer sentiment patterns and trends across time periods")
        
        # Create sentiment trend analysis
        trend_fig = visualizer.create_sentiment_trend_analysis(feedback_data)
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
        st.header("Actionable Insights Analysis")
        st.markdown("Visual representation of actionable items and priority analysis")
        
        # Create actionable insights analysis
        insights_fig = visualizer.create_actionable_insights_analysis(feedback_data)
        if insights_fig:
            st.plotly_chart(insights_fig, use_container_width=True)
        
        # Add actionable insights
        st.subheader("🎯 Actionable Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bug_actions = sum(1 for actions in feedback_data['actionable_items'] 
                            for action in actions if action[0] == 'bug')
            st.metric(
                "Bug Reports", 
                f"{bug_actions:,}",
                delta="12.3%"
            )
        
        with col2:
            feature_requests = sum(1 for actions in feedback_data['actionable_items'] 
                                 for action in actions if action[0] == 'feature request')
            st.metric(
                "Feature Requests", 
                f"{feature_requests:,}",
                delta="25.7%"
            )
        
        with col3:
            ui_issues = sum(1 for actions in feedback_data['actionable_items'] 
                          for action in actions if action[0] == 'ui/ux')
            st.metric(
                "UI/UX Issues", 
                f"{ui_issues:,}",
                delta="8.1%"
            )
    
    with tab4:
        st.header("Performance Metrics")
        st.markdown("Key performance indicators and customer service metrics")
        
        # Create performance metrics
        metrics_fig = visualizer.create_performance_metrics(feedback_data)
        if metrics_fig:
            st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Add performance insights
        st.subheader("📈 Performance Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **Strong Performance**:
            - High customer satisfaction rate
            - Good sentiment distribution
            - Efficient feedback processing
            """)
        
        with col2:
            st.info("""
            **Optimization Opportunities**:
            - Reduce complaint resolution time
            - Improve response quality
            - Enhance proactive communication
            """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Customer Service Feedback Analysis Tool - Comprehensive Dashboard</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    create_comprehensive_customer_service_dashboard()
