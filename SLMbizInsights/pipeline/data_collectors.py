"""
Data collection modules for internal and external data sources
"""
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

class DataCollector:
    """Base class for data collectors"""
    
    def __init__(self, api_url: str, headers: Optional[Dict] = None):
        self.api_url = api_url
        self.headers = headers or {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def fetch_data(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch data from API endpoint"""
        try:
            url = f"{self.api_url}/{endpoint}"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching data from {endpoint}: {e}")
            return None

class CRMDataCollector(DataCollector):
    """Collects CRM data including customer demographics, purchase history, and feedback"""
    
    def __init__(self):
        super().__init__(Config.CRM_API_URL)
    
    def get_customers(self, limit: int = 1000) -> pd.DataFrame:
        """Fetch customer data"""
        data = self.fetch_data("customers", {"limit": limit})
        if data:
            return pd.DataFrame(data.get('customers', []))
        return pd.DataFrame()
    
    def get_purchase_history(self, customer_id: Optional[str] = None) -> pd.DataFrame:
        """Fetch purchase history"""
        params = {}
        if customer_id:
            params['customer_id'] = customer_id
        
        data = self.fetch_data("purchases", params)
        if data:
            return pd.DataFrame(data.get('purchases', []))
        return pd.DataFrame()
    
    def get_customer_feedback(self, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch customer feedback"""
        params = {}
        if start_date:
            params['start_date'] = start_date.isoformat()
        
        data = self.fetch_data("feedback", params)
        if data:
            return pd.DataFrame(data.get('feedback', []))
        return pd.DataFrame()

class SalesDataCollector(DataCollector):
    """Collects sales data including transaction records and sales reports"""
    
    def __init__(self):
        super().__init__(Config.SALES_API_URL)
    
    def get_sales_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch sales data for date range"""
        params = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }
        
        data = self.fetch_data("sales", params)
        if data:
            return pd.DataFrame(data.get('sales', []))
        return pd.DataFrame()
    
    def get_product_performance(self) -> pd.DataFrame:
        """Fetch product performance data"""
        data = self.fetch_data("products/performance")
        if data:
            return pd.DataFrame(data.get('products', []))
        return pd.DataFrame()
    
    def get_sales_reports(self, report_type: str = "monthly") -> pd.DataFrame:
        """Fetch sales reports"""
        data = self.fetch_data(f"reports/{report_type}")
        if data:
            return pd.DataFrame(data.get('reports', []))
        return pd.DataFrame()

class FinancialDataCollector(DataCollector):
    """Collects financial data including revenue, costs, and profit margins"""
    
    def __init__(self):
        super().__init__(Config.FINANCIAL_API_URL)
    
    def get_revenue_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch revenue data"""
        params = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }
        
        data = self.fetch_data("revenue", params)
        if data:
            return pd.DataFrame(data.get('revenue', []))
        return pd.DataFrame()
    
    def get_cost_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch cost data"""
        params = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }
        
        data = self.fetch_data("costs", params)
        if data:
            return pd.DataFrame(data.get('costs', []))
        return pd.DataFrame()
    
    def get_profit_margins(self) -> pd.DataFrame:
        """Fetch profit margin data"""
        data = self.fetch_data("profit-margins")
        if data:
            return pd.DataFrame(data.get('margins', []))
        return pd.DataFrame()

class ExternalDataCollector:
    """Collects external data from market trends, economic indicators, etc."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'SLM-BI/1.0',
            'Accept': 'application/json'
        }
    
    def get_market_trends(self, industry: str) -> pd.DataFrame:
        """Fetch market trend data (mock implementation)"""
        # In a real implementation, this would connect to market data APIs
        # For now, we'll return sample data
        sample_data = {
            'date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
            'industry': [industry] * 365,
            'growth_rate': [0.02 + 0.01 * (i % 30) / 30 for i in range(365)],
            'market_size': [1000000 + i * 1000 for i in range(365)]
        }
        return pd.DataFrame(sample_data)
    
    def get_economic_indicators(self) -> pd.DataFrame:
        """Fetch economic indicators (mock implementation)"""
        sample_data = {
            'date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='M'),
            'inflation_rate': [0.03 + 0.01 * (i % 12) / 12 for i in range(12)],
            'gdp_growth': [0.025 + 0.005 * (i % 12) / 12 for i in range(12)],
            'consumer_confidence': [100 + 5 * (i % 12) for i in range(12)]
        }
        return pd.DataFrame(sample_data)
    
    def get_competitor_analysis(self, competitors: List[str]) -> pd.DataFrame:
        """Fetch competitor analysis data (mock implementation)"""
        data = []
        for competitor in competitors:
            for i in range(12):  # 12 months
                data.append({
                    'competitor': competitor,
                    'month': f"2023-{i+1:02d}",
                    'market_share': 0.1 + 0.05 * (i % 6) / 6,
                    'revenue': 1000000 + i * 50000,
                    'growth_rate': 0.02 + 0.01 * (i % 4) / 4
                })
        return pd.DataFrame(data)

class DataCollectionOrchestrator:
    """Orchestrates data collection from all sources"""
    
    def __init__(self):
        self.crm_collector = CRMDataCollector()
        self.sales_collector = SalesDataCollector()
        self.financial_collector = FinancialDataCollector()
        self.external_collector = ExternalDataCollector()
    
    def collect_all_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Collect data from all sources"""
        logger.info(f"Starting data collection from {start_date} to {end_date}")
        
        data = {}
        
        try:
            # Internal data
            data['customers'] = self.crm_collector.get_customers()
            data['purchases'] = self.crm_collector.get_purchase_history()
            data['feedback'] = self.crm_collector.get_customer_feedback(start_date)
            data['sales'] = self.sales_collector.get_sales_data(start_date, end_date)
            data['products'] = self.sales_collector.get_product_performance()
            data['revenue'] = self.financial_collector.get_revenue_data(start_date, end_date)
            data['costs'] = self.financial_collector.get_cost_data(start_date, end_date)
            data['profit_margins'] = self.financial_collector.get_profit_margins()
            
            # External data
            data['market_trends'] = self.external_collector.get_market_trends("technology")
            data['economic_indicators'] = self.external_collector.get_economic_indicators()
            data['competitors'] = self.external_collector.get_competitor_analysis(
                ["Competitor A", "Competitor B", "Competitor C"]
            )
            
            logger.info("Data collection completed successfully")
            
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
        
        return data
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = Config.RAW_DATA_DIR):
        """Save collected data to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data.items():
            if not df.empty:
                filepath = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {name} data to {filepath}")

# Sample data generator for testing
class SampleDataGenerator:
    """Generates sample data for testing and development"""
    
    @staticmethod
    def generate_customer_data(n_customers: int = 1000) -> pd.DataFrame:
        """Generate sample customer data"""
        import random
        from faker import Faker
        
        fake = Faker()
        data = []
        
        for i in range(n_customers):
            data.append({
                'customer_id': f"CUST_{i:06d}",
                'name': fake.name(),
                'email': fake.email(),
                'age': random.randint(18, 80),
                'gender': random.choice(['M', 'F', 'Other']),
                'location': fake.city(),
                'registration_date': fake.date_between(start_date='-2y', end_date='today'),
                'total_spend': random.uniform(100, 10000),
                'retention_rate': random.uniform(0.1, 0.9),
                'segment': random.choice(['Premium', 'Standard', 'Basic'])
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_sales_data(n_transactions: int = 5000) -> pd.DataFrame:
        """Generate sample sales data"""
        import random
        from faker import Faker
        
        fake = Faker()
        data = []
        
        for i in range(n_transactions):
            data.append({
                'transaction_id': f"TXN_{i:08d}",
                'customer_id': f"CUST_{random.randint(0, 999):06d}",
                'product_id': f"PROD_{random.randint(1, 100):03d}",
                'amount': random.uniform(10, 1000),
                'quantity': random.randint(1, 10),
                'date': fake.date_between(start_date='-1y', end_date='today'),
                'region': random.choice(['North', 'South', 'East', 'West']),
                'channel': random.choice(['Online', 'Store', 'Phone'])
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_feedback_data(n_feedback: int = 2000) -> pd.DataFrame:
        """Generate sample feedback data"""
        import random
        from faker import Faker
        
        fake = Faker()
        data = []
        
        feedback_templates = [
            "Great product, very satisfied with the quality",
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
        
        for i in range(n_feedback):
            data.append({
                'feedback_id': f"FB_{i:06d}",
                'customer_id': f"CUST_{random.randint(0, 999):06d}",
                'product_id': f"PROD_{random.randint(1, 100):03d}",
                'rating': random.randint(1, 5),
                'text': random.choice(feedback_templates),
                'date': fake.date_between(start_date='-1y', end_date='today'),
                'category': random.choice(['Product', 'Service', 'Delivery', 'Support'])
            })
        
        return pd.DataFrame(data)
