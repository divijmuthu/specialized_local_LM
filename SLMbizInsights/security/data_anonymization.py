"""
Data anonymization and privacy protection module
"""
import pandas as pd
import numpy as np
import hashlib
import uuid
import re
from typing import Dict, List, Optional, Any, Union
import logging
from cryptography.fernet import Fernet
import bcrypt
import json
import os
from datetime import datetime, timedelta
import random
import string

logger = logging.getLogger(__name__)

class DataAnonymizer:
    """Data anonymization utilities"""
    
    def __init__(self):
        self.encryption_key = None
        self.salt_rounds = 12
    
    def generate_encryption_key(self) -> bytes:
        """Generate encryption key for data protection"""
        self.encryption_key = Fernet.generate_key()
        return self.encryption_key
    
    def load_encryption_key(self, key_path: str):
        """Load encryption key from file"""
        try:
            with open(key_path, 'rb') as f:
                self.encryption_key = f.read()
        except FileNotFoundError:
            logger.warning(f"Encryption key file not found: {key_path}")
            self.generate_encryption_key()
    
    def save_encryption_key(self, key_path: str):
        """Save encryption key to file"""
        if self.encryption_key is None:
            self.generate_encryption_key()
        
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, 'wb') as f:
            f.write(self.encryption_key)
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data using bcrypt"""
        if salt is None:
            salt = bcrypt.gensalt(rounds=self.salt_rounds)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        hashed = bcrypt.hashpw(data, salt)
        return hashed.decode('utf-8')
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if self.encryption_key is None:
            self.generate_encryption_key()
        
        fernet = Fernet(self.encryption_key)
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = fernet.encrypt(data)
        return encrypted.decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if self.encryption_key is None:
            raise ValueError("Encryption key not loaded")
        
        fernet = Fernet(self.encryption_key)
        decrypted = fernet.decrypt(encrypted_data.encode('utf-8'))
        return decrypted.decode('utf-8')
    
    def anonymize_customer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymize customer data"""
        anonymized_df = df.copy()
        
        # Anonymize customer IDs
        if 'customer_id' in anonymized_df.columns:
            customer_mapping = {}
            for idx, customer_id in enumerate(anonymized_df['customer_id'].unique()):
                customer_mapping[customer_id] = f"CUST_{idx:06d}"
            anonymized_df['customer_id'] = anonymized_df['customer_id'].map(customer_mapping)
        
        # Anonymize names
        if 'name' in anonymized_df.columns:
            anonymized_df['name'] = anonymized_df['name'].apply(self._anonymize_name)
        
        # Anonymize emails
        if 'email' in anonymized_df.columns:
            anonymized_df['email'] = anonymized_df['email'].apply(self._anonymize_email)
        
        # Anonymize phone numbers
        if 'phone' in anonymized_df.columns:
            anonymized_df['phone'] = anonymized_df['phone'].apply(self._anonymize_phone)
        
        # Anonymize addresses
        if 'address' in anonymized_df.columns:
            anonymized_df['address'] = anonymized_df['address'].apply(self._anonymize_address)
        
        # Generalize age (group into ranges)
        if 'age' in anonymized_df.columns:
            anonymized_df['age'] = anonymized_df['age'].apply(self._generalize_age)
        
        # Generalize location (city level only)
        if 'location' in anonymized_df.columns:
            anonymized_df['location'] = anonymized_df['location'].apply(self._generalize_location)
        
        return anonymized_df
    
    def _anonymize_name(self, name: str) -> str:
        """Anonymize name while preserving format"""
        if pd.isna(name) or name == '':
            return name
        
        # Generate random name with same format
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Emily']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _anonymize_email(self, email: str) -> str:
        """Anonymize email while preserving format"""
        if pd.isna(email) or email == '':
            return email
        
        # Extract domain
        if '@' in email:
            domain = email.split('@')[1]
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            return f"{username}@{domain}"
        
        return email
    
    def _anonymize_phone(self, phone: str) -> str:
        """Anonymize phone number while preserving format"""
        if pd.isna(phone) or phone == '':
            return phone
        
        # Keep format but randomize digits
        digits = re.findall(r'\d', phone)
        if len(digits) >= 10:
            # Generate random phone number with same format
            area_code = ''.join(random.choices(string.digits, k=3))
            exchange = ''.join(random.choices(string.digits, k=3))
            number = ''.join(random.choices(string.digits, k=4))
            return f"({area_code}) {exchange}-{number}"
        
        return phone
    
    def _anonymize_address(self, address: str) -> str:
        """Anonymize address while preserving format"""
        if pd.isna(address) or address == '':
            return address
        
        # Generate random address
        street_numbers = [str(random.randint(100, 9999))]
        street_names = ['Main St', 'Oak Ave', 'Pine Rd', 'Cedar Ln', 'Maple Dr']
        cities = ['Anytown', 'Somewhere', 'Nowhere', 'Everywhere', 'Anywhere']
        states = ['CA', 'NY', 'TX', 'FL', 'IL']
        
        street_number = random.choice(street_numbers)
        street_name = random.choice(street_names)
        city = random.choice(cities)
        state = random.choice(states)
        zip_code = ''.join(random.choices(string.digits, k=5))
        
        return f"{street_number} {street_name}, {city}, {state} {zip_code}"
    
    def _generalize_age(self, age: Union[int, float]) -> str:
        """Generalize age into ranges"""
        if pd.isna(age):
            return 'Unknown'
        
        age = int(age)
        if age < 18:
            return 'Under 18'
        elif age < 25:
            return '18-24'
        elif age < 35:
            return '25-34'
        elif age < 45:
            return '35-44'
        elif age < 55:
            return '45-54'
        elif age < 65:
            return '55-64'
        else:
            return '65+'
    
    def _generalize_location(self, location: str) -> str:
        """Generalize location to city level"""
        if pd.isna(location) or location == '':
            return 'Unknown'
        
        # Extract city from location string
        parts = location.split(',')
        if len(parts) > 0:
            return parts[0].strip()
        
        return location
    
    def anonymize_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymize financial data"""
        anonymized_df = df.copy()
        
        # Anonymize transaction IDs
        if 'transaction_id' in anonymized_df.columns:
            transaction_mapping = {}
            for idx, trans_id in enumerate(anonymized_df['transaction_id'].unique()):
                transaction_mapping[trans_id] = f"TXN_{idx:08d}"
            anonymized_df['transaction_id'] = anonymized_df['transaction_id'].map(transaction_mapping)
        
        # Generalize amounts (round to nearest 10 or 100)
        amount_columns = ['amount', 'price', 'cost', 'revenue', 'profit']
        for col in amount_columns:
            if col in anonymized_df.columns:
                anonymized_df[col] = anonymized_df[col].apply(self._generalize_amount)
        
        # Anonymize account numbers
        if 'account_number' in anonymized_df.columns:
            anonymized_df['account_number'] = anonymized_df['account_number'].apply(self._anonymize_account_number)
        
        return anonymized_df
    
    def _generalize_amount(self, amount: Union[int, float]) -> float:
        """Generalize financial amounts"""
        if pd.isna(amount):
            return amount
        
        amount = float(amount)
        if amount < 100:
            return round(amount, -1)  # Round to nearest 10
        elif amount < 1000:
            return round(amount, -2)  # Round to nearest 100
        else:
            return round(amount, -3)  # Round to nearest 1000
    
    def _anonymize_account_number(self, account_number: str) -> str:
        """Anonymize account numbers"""
        if pd.isna(account_number) or account_number == '':
            return account_number
        
        # Keep first 4 and last 4 digits, mask the middle
        if len(account_number) >= 8:
            return f"{account_number[:4]}****{account_number[-4:]}"
        
        return '****'
    
    def anonymize_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymize text data"""
        anonymized_df = df.copy()
        
        # Anonymize text fields
        text_columns = ['text', 'description', 'comments', 'notes', 'feedback']
        for col in text_columns:
            if col in anonymized_df.columns:
                anonymized_df[col] = anonymized_df[col].apply(self._anonymize_text)
        
        return anonymized_df
    
    def _anonymize_text(self, text: str) -> str:
        """Anonymize text while preserving structure"""
        if pd.isna(text) or text == '':
            return text
        
        # Remove or replace sensitive information
        # Replace email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Replace phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Replace credit card numbers
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        
        # Replace SSN
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        return text
    
    def create_synthetic_data(self, original_df: pd.DataFrame, 
                            preserve_structure: bool = True) -> pd.DataFrame:
        """Create synthetic data based on original data structure"""
        synthetic_df = pd.DataFrame()
        
        for column in original_df.columns:
            if original_df[column].dtype == 'object':
                # Categorical data - sample from unique values
                unique_values = original_df[column].dropna().unique()
                if len(unique_values) > 0:
                    synthetic_df[column] = np.random.choice(unique_values, size=len(original_df))
                else:
                    synthetic_df[column] = np.nan
            elif original_df[column].dtype in ['int64', 'float64']:
                # Numeric data - sample from distribution
                if preserve_structure:
                    # Preserve statistical properties
                    mean = original_df[column].mean()
                    std = original_df[column].std()
                    synthetic_df[column] = np.random.normal(mean, std, size=len(original_df))
                else:
                    # Random values in same range
                    min_val = original_df[column].min()
                    max_val = original_df[column].max()
                    synthetic_df[column] = np.random.uniform(min_val, max_val, size=len(original_df))
            else:
                # Other data types - copy structure
                synthetic_df[column] = original_df[column].copy()
        
        return synthetic_df

class PrivacyCompliance:
    """Privacy compliance utilities"""
    
    def __init__(self):
        self.gdpr_requirements = {
            'data_minimization': True,
            'purpose_limitation': True,
            'storage_limitation': True,
            'accuracy': True,
            'security': True,
            'accountability': True
        }
    
    def check_gdpr_compliance(self, df: pd.DataFrame, 
                            data_types: Dict[str, str]) -> Dict[str, Any]:
        """Check GDPR compliance of data"""
        compliance_report = {
            'compliant': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check for personal data
        personal_data_columns = []
        for column, data_type in data_types.items():
            if data_type in ['name', 'email', 'phone', 'address', 'ssn', 'credit_card']:
                personal_data_columns.append(column)
        
        if personal_data_columns:
            compliance_report['issues'].append(f"Personal data found in columns: {personal_data_columns}")
            compliance_report['recommendations'].append("Anonymize or pseudonymize personal data")
            compliance_report['compliant'] = False
        
        # Check data retention
        if 'date' in df.columns or 'created_at' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'created_at'
            max_date = pd.to_datetime(df[date_col]).max()
            days_old = (datetime.now() - max_date).days
            
            if days_old > 2555:  # 7 years
                compliance_report['issues'].append("Data older than 7 years found")
                compliance_report['recommendations'].append("Review data retention policy")
        
        # Check data accuracy
        null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if null_percentage > 20:
            compliance_report['issues'].append(f"High percentage of missing data: {null_percentage:.1f}%")
            compliance_report['recommendations'].append("Improve data quality")
        
        return compliance_report
    
    def generate_privacy_policy(self, data_types: Dict[str, str]) -> str:
        """Generate privacy policy based on data types"""
        policy = """
        PRIVACY POLICY
        
        Data Collection:
        We collect the following types of data:
        """
        
        for column, data_type in data_types.items():
            if data_type in ['name', 'email', 'phone', 'address']:
                policy += f"- {column}: Personal identification information\n"
            elif data_type in ['age', 'gender', 'location']:
                policy += f"- {column}: Demographic information\n"
            elif data_type in ['purchase_history', 'preferences']:
                policy += f"- {column}: Behavioral data\n"
            else:
                policy += f"- {column}: {data_type} data\n"
        
        policy += """
        
        Data Usage:
        - Business analytics and insights
        - Customer segmentation
        - Product recommendations
        - Service improvement
        
        Data Protection:
        - All personal data is anonymized
        - Data is encrypted in transit and at rest
        - Access is restricted to authorized personnel
        - Regular security audits are conducted
        
        Your Rights:
        - Right to access your data
        - Right to rectification
        - Right to erasure
        - Right to data portability
        - Right to object to processing
        
        Contact:
        For privacy-related inquiries, contact: privacy@company.com
        """
        
        return policy
    
    def create_data_inventory(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create data inventory for compliance"""
        inventory = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'data_types': df.dtypes.to_dict(),
            'personal_data': [],
            'sensitive_data': [],
            'business_data': [],
            'retention_period': '7 years',
            'last_updated': datetime.now().isoformat()
        }
        
        # Categorize data
        for column in df.columns:
            if any(keyword in column.lower() for keyword in ['name', 'email', 'phone', 'address', 'ssn']):
                inventory['personal_data'].append(column)
            elif any(keyword in column.lower() for keyword in ['credit', 'card', 'bank', 'account', 'salary']):
                inventory['sensitive_data'].append(column)
            else:
                inventory['business_data'].append(column)
        
        return inventory

class SecurityManager:
    """Security management utilities"""
    
    def __init__(self):
        self.access_logs = []
        self.security_events = []
    
    def log_access(self, user_id: str, resource: str, action: str, 
                  timestamp: Optional[datetime] = None):
        """Log data access for audit trail"""
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'ip_address': '127.0.0.1',  # Placeholder
            'user_agent': 'SLM-BI/1.0'
        }
        
        self.access_logs.append(log_entry)
        logger.info(f"Access logged: {user_id} {action} {resource}")
    
    def log_security_event(self, event_type: str, description: str, 
                          severity: str = 'medium', user_id: Optional[str] = None):
        """Log security events"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'description': description,
            'severity': severity,
            'user_id': user_id
        }
        
        self.security_events.append(event)
        logger.warning(f"Security event: {event_type} - {description}")
    
    def check_access_permissions(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has permission to access resource"""
        # Simple permission check - in production, use proper RBAC
        admin_users = ['admin', 'data_scientist', 'analyst']
        
        if user_id in admin_users:
            return True
        
        # Check resource-specific permissions
        if 'sensitive' in resource.lower() and user_id not in admin_users:
            self.log_security_event('unauthorized_access_attempt', 
                                  f"User {user_id} attempted to access {resource}")
            return False
        
        return True
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate security report"""
        report = {
            'total_access_logs': len(self.access_logs),
            'total_security_events': len(self.security_events),
            'recent_events': self.security_events[-10:] if self.security_events else [],
            'high_severity_events': [e for e in self.security_events if e['severity'] == 'high'],
            'unauthorized_access_attempts': [e for e in self.security_events 
                                           if e['event_type'] == 'unauthorized_access_attempt']
        }
        
        return report

if __name__ == "__main__":
    # Test the security and privacy modules
    anonymizer = DataAnonymizer()
    compliance = PrivacyCompliance()
    security = SecurityManager()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
        'phone': ['(555) 123-4567', '(555) 987-6543', '(555) 456-7890'],
        'age': [25, 35, 45],
        'amount': [100.50, 250.75, 75.25]
    })
    
    # Test anonymization
    anonymized_data = anonymizer.anonymize_customer_data(sample_data)
    print("Anonymized data:")
    print(anonymized_data)
    
    # Test compliance
    data_types = {
        'customer_id': 'identifier',
        'name': 'name',
        'email': 'email',
        'phone': 'phone',
        'age': 'age',
        'amount': 'amount'
    }
    
    compliance_report = compliance.check_gdpr_compliance(sample_data, data_types)
    print("\nCompliance report:")
    print(compliance_report)
    
    # Test security
    security.log_access('user1', 'customer_data', 'read')
    security.log_security_event('data_access', 'User accessed customer data')
    
    security_report = security.generate_security_report()
    print("\nSecurity report:")
    print(security_report)
    
    print("Security and privacy modules test completed successfully!")
