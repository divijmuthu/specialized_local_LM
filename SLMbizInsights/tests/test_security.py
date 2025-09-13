"""
Tests for security and privacy components
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.data_anonymization import (
    DataAnonymizer, PrivacyCompliance, SecurityManager
)

class TestDataAnonymizer:
    """Test data anonymization components"""
    
    def test_data_anonymizer_init(self):
        """Test DataAnonymizer initialization"""
        anonymizer = DataAnonymizer()
        assert anonymizer.encryption_key is None
        assert anonymizer.salt_rounds == 12
    
    def test_generate_encryption_key(self):
        """Test encryption key generation"""
        anonymizer = DataAnonymizer()
        key = anonymizer.generate_encryption_key()
        assert key is not None
        assert len(key) > 0
        assert anonymizer.encryption_key == key
    
    def test_hash_data(self):
        """Test data hashing"""
        anonymizer = DataAnonymizer()
        
        test_data = "sensitive information"
        hashed = anonymizer.hash_data(test_data)
        
        assert hashed != test_data
        assert len(hashed) > 0
        assert isinstance(hashed, str)
    
    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption"""
        anonymizer = DataAnonymizer()
        anonymizer.generate_encryption_key()
        
        test_data = "sensitive information"
        encrypted = anonymizer.encrypt_data(test_data)
        decrypted = anonymizer.decrypt_data(encrypted)
        
        assert encrypted != test_data
        assert decrypted == test_data
    
    def test_anonymize_customer_data(self):
        """Test customer data anonymization"""
        anonymizer = DataAnonymizer()
        
        test_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
            'phone': ['(555) 123-4567', '(555) 987-6543', '(555) 456-7890'],
            'age': [25, 35, 45],
            'location': ['New York, NY', 'Los Angeles, CA', 'Chicago, IL']
        })
        
        anonymized = anonymizer.anonymize_customer_data(test_data)
        
        # Check that sensitive data is anonymized
        assert anonymized['customer_id'].iloc[0] != 'CUST_001'
        assert anonymized['name'].iloc[0] != 'John Doe'
        assert anonymized['email'].iloc[0] != 'john@example.com'
        assert anonymized['phone'].iloc[0] != '(555) 123-4567'
        
        # Check that age is generalized
        assert anonymized['age'].iloc[0] in ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        
        # Check that location is generalized
        assert anonymized['location'].iloc[0] in ['New York', 'Los Angeles', 'Chicago']
    
    def test_anonymize_financial_data(self):
        """Test financial data anonymization"""
        anonymizer = DataAnonymizer()
        
        test_data = pd.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003'],
            'amount': [123.45, 678.90, 234.56],
            'account_number': ['1234567890', '0987654321', '1122334455']
        })
        
        anonymized = anonymizer.anonymize_financial_data(test_data)
        
        # Check that transaction IDs are anonymized
        assert anonymized['transaction_id'].iloc[0] != 'TXN_001'
        
        # Check that amounts are generalized
        assert anonymized['amount'].iloc[0] in [120, 130, 140, 150, 160, 170, 180, 190, 200]
        
        # Check that account numbers are masked
        assert anonymized['account_number'].iloc[0].startswith('1234****')
    
    def test_anonymize_text_data(self):
        """Test text data anonymization"""
        anonymizer = DataAnonymizer()
        
        test_data = pd.DataFrame({
            'text': [
                'Contact me at john@example.com or call (555) 123-4567',
                'My credit card is 1234-5678-9012-3456',
                'My SSN is 123-45-6789'
            ]
        })
        
        anonymized = anonymizer.anonymize_text_data(test_data)
        
        # Check that sensitive information is replaced
        assert '[EMAIL]' in anonymized['text'].iloc[0]
        assert '[PHONE]' in anonymized['text'].iloc[0]
        assert '[CARD]' in anonymized['text'].iloc[1]
        assert '[SSN]' in anonymized['text'].iloc[2]
    
    def test_create_synthetic_data(self):
        """Test synthetic data creation"""
        anonymizer = DataAnonymizer()
        
        original_data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'value': [10, 20, 15, 25],
            'score': [0.8, 0.9, 0.7, 0.85]
        })
        
        synthetic_data = anonymizer.create_synthetic_data(original_data)
        
        # Check that structure is preserved
        assert len(synthetic_data) == len(original_data)
        assert list(synthetic_data.columns) == list(original_data.columns)
        
        # Check that categorical data is sampled from original values
        assert all(cat in original_data['category'].values for cat in synthetic_data['category'].values)
        
        # Check that numeric data follows similar distribution
        assert synthetic_data['value'].min() >= original_data['value'].min()
        assert synthetic_data['value'].max() <= original_data['value'].max()

class TestPrivacyCompliance:
    """Test privacy compliance components"""
    
    def test_privacy_compliance_init(self):
        """Test PrivacyCompliance initialization"""
        compliance = PrivacyCompliance()
        assert compliance.gdpr_requirements is not None
        assert 'data_minimization' in compliance.gdpr_requirements
    
    def test_check_gdpr_compliance(self):
        """Test GDPR compliance checking"""
        compliance = PrivacyCompliance()
        
        test_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'name': ['John Doe', 'Jane Smith'],
            'email': ['john@example.com', 'jane@example.com'],
            'age': [25, 35],
            'purchase_amount': [100, 200]
        })
        
        data_types = {
            'customer_id': 'identifier',
            'name': 'name',
            'email': 'email',
            'age': 'age',
            'purchase_amount': 'amount'
        }
        
        report = compliance.check_gdpr_compliance(test_data, data_types)
        
        assert 'compliant' in report
        assert 'issues' in report
        assert 'recommendations' in report
        
        # Should detect personal data
        assert not report['compliant']
        assert len(report['issues']) > 0
        assert len(report['recommendations']) > 0
    
    def test_check_gdpr_compliance_clean_data(self):
        """Test GDPR compliance with clean data"""
        compliance = PrivacyCompliance()
        
        test_data = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_002'],
            'category': ['Electronics', 'Clothing'],
            'price': [100, 50],
            'sales_count': [10, 20]
        })
        
        data_types = {
            'product_id': 'identifier',
            'category': 'category',
            'price': 'amount',
            'sales_count': 'count'
        }
        
        report = compliance.check_gdpr_compliance(test_data, data_types)
        
        # Should be compliant as no personal data
        assert report['compliant']
        assert len(report['issues']) == 0
    
    def test_generate_privacy_policy(self):
        """Test privacy policy generation"""
        compliance = PrivacyCompliance()
        
        data_types = {
            'name': 'name',
            'email': 'email',
            'age': 'age',
            'purchase_history': 'purchase_history'
        }
        
        policy = compliance.generate_privacy_policy(data_types)
        
        assert 'PRIVACY POLICY' in policy
        assert 'Data Collection:' in policy
        assert 'Data Usage:' in policy
        assert 'Data Protection:' in policy
        assert 'Your Rights:' in policy
    
    def test_create_data_inventory(self):
        """Test data inventory creation"""
        compliance = PrivacyCompliance()
        
        test_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'name': ['John Doe', 'Jane Smith'],
            'email': ['john@example.com', 'jane@example.com'],
            'credit_card': ['1234-5678-9012-3456', '0987-6543-2109-8765'],
            'purchase_amount': [100, 200],
            'product_category': ['Electronics', 'Clothing']
        })
        
        inventory = compliance.create_data_inventory(test_data)
        
        assert 'total_records' in inventory
        assert 'total_columns' in inventory
        assert 'personal_data' in inventory
        assert 'sensitive_data' in inventory
        assert 'business_data' in inventory
        
        assert inventory['total_records'] == 2
        assert inventory['total_columns'] == 6
        assert 'name' in inventory['personal_data']
        assert 'email' in inventory['personal_data']
        assert 'credit_card' in inventory['sensitive_data']
        assert 'purchase_amount' in inventory['business_data']

class TestSecurityManager:
    """Test security management components"""
    
    def test_security_manager_init(self):
        """Test SecurityManager initialization"""
        security = SecurityManager()
        assert security.access_logs == []
        assert security.security_events == []
    
    def test_log_access(self):
        """Test access logging"""
        security = SecurityManager()
        
        security.log_access('user1', 'customer_data', 'read')
        
        assert len(security.access_logs) == 1
        log_entry = security.access_logs[0]
        assert log_entry['user_id'] == 'user1'
        assert log_entry['resource'] == 'customer_data'
        assert log_entry['action'] == 'read'
        assert 'timestamp' in log_entry
    
    def test_log_security_event(self):
        """Test security event logging"""
        security = SecurityManager()
        
        security.log_security_event('unauthorized_access', 'User attempted unauthorized access', 'high', 'user1')
        
        assert len(security.security_events) == 1
        event = security.security_events[0]
        assert event['event_type'] == 'unauthorized_access'
        assert event['description'] == 'User attempted unauthorized access'
        assert event['severity'] == 'high'
        assert event['user_id'] == 'user1'
    
    def test_check_access_permissions(self):
        """Test access permission checking"""
        security = SecurityManager()
        
        # Test admin user
        assert security.check_access_permissions('admin', 'sensitive_data', 'read') == True
        
        # Test regular user with sensitive data
        assert security.check_access_permissions('user1', 'sensitive_data', 'read') == False
        
        # Test regular user with normal data
        assert security.check_access_permissions('user1', 'normal_data', 'read') == True
        
        # Check that unauthorized access is logged
        assert len(security.security_events) > 0
        assert security.security_events[-1]['event_type'] == 'unauthorized_access_attempt'
    
    def test_generate_security_report(self):
        """Test security report generation"""
        security = SecurityManager()
        
        # Add some test data
        security.log_access('user1', 'data1', 'read')
        security.log_access('user2', 'data2', 'write')
        security.log_security_event('test_event', 'Test security event', 'medium')
        
        report = security.generate_security_report()
        
        assert 'total_access_logs' in report
        assert 'total_security_events' in report
        assert 'recent_events' in report
        assert 'high_severity_events' in report
        assert 'unauthorized_access_attempts' in report
        
        assert report['total_access_logs'] == 2
        assert report['total_security_events'] == 1
        assert len(report['recent_events']) == 1

class TestSecurityIntegration:
    """Test security components integration"""
    
    def test_end_to_end_security_workflow(self):
        """Test complete security workflow"""
        # Initialize components
        anonymizer = DataAnonymizer()
        compliance = PrivacyCompliance()
        security = SecurityManager()
        
        # Create test data
        test_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'name': ['John Doe', 'Jane Smith'],
            'email': ['john@example.com', 'jane@example.com'],
            'phone': ['(555) 123-4567', '(555) 987-6543'],
            'amount': [100.50, 200.75]
        })
        
        # Check compliance
        data_types = {
            'customer_id': 'identifier',
            'name': 'name',
            'email': 'email',
            'phone': 'phone',
            'amount': 'amount'
        }
        
        compliance_report = compliance.check_gdpr_compliance(test_data, data_types)
        assert not compliance_report['compliant']  # Should fail due to personal data
        
        # Anonymize data
        anonymized_data = anonymizer.anonymize_customer_data(test_data)
        
        # Check compliance again
        anonymized_data_types = {
            'customer_id': 'identifier',
            'name': 'name',
            'email': 'email',
            'phone': 'phone',
            'amount': 'amount'
        }
        
        compliance_report_after = compliance.check_gdpr_compliance(anonymized_data, anonymized_data_types)
        # Should still fail because we're checking the same data types, but data is anonymized
        
        # Log access
        security.log_access('analyst1', 'customer_data', 'read')
        
        # Check permissions
        has_permission = security.check_access_permissions('analyst1', 'customer_data', 'read')
        assert has_permission
        
        # Generate reports
        security_report = security.generate_security_report()
        inventory = compliance.create_data_inventory(test_data)
        
        assert len(security_report['total_access_logs']) >= 1
        assert inventory['total_records'] == 2

if __name__ == "__main__":
    pytest.main([__file__])
