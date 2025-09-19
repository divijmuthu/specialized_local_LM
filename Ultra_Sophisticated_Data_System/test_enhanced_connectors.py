"""
Test suite for enhanced data connectors and advanced features
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import tempfile
import os
import time
from datetime import datetime, timedelta

# Import the system components
import sys
import importlib.util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load the module
spec = importlib.util.spec_from_file_location('palantir_advanced_system', 'palantir-advanced-system.py')
palantir_advanced_system = importlib.util.module_from_spec(spec)
spec.loader.exec_module(palantir_advanced_system)

# Import components
DataFusionEngine = palantir_advanced_system.DataFusionEngine
MongoDBDataConnector = palantir_advanced_system.MongoDBDataConnector
RedisDataConnector = palantir_advanced_system.RedisDataConnector
StreamingDataConnector = palantir_advanced_system.StreamingDataConnector
EnhancedInsightGenerator = palantir_advanced_system.EnhancedInsightGenerator
AdvancedDataSourceConfig = palantir_advanced_system.AdvancedDataSourceConfig
DataDomain = palantir_advanced_system.DataDomain
DataSecurityLevel = palantir_advanced_system.DataSecurityLevel


class TestEnhancedDataConnectors:
    """Test suite for enhanced data connectors"""
    
    @pytest.mark.asyncio
    async def test_mongodb_connector(self):
        """Test MongoDB connector functionality"""
        print("\n🍃 Testing MongoDB Connector")
        
        config = AdvancedDataSourceConfig(
            source_type=DataDomain.GENERIC,
            connection_params={
                'host': 'localhost',
                'port': 27017,
                'database': 'test_db',
                'collection': 'test_collection'
            },
            data_domain=DataDomain.GENERIC,
            security_level=DataSecurityLevel.INTERNAL
        )
        
        connector = MongoDBDataConnector(config)
        
        # Test connection
        success = await connector.connect()
        assert success is True
        print(f"   ✅ MongoDB connection successful")
        
        # Test data extraction
        data = await connector.extract()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'document_id' in data.columns
        assert 'user_id' in data.columns
        assert 'event_type' in data.columns
        print(f"   ✅ MongoDB data extraction: {len(data)} documents")
        
        # Test schema validation
        is_valid, schema = await connector.validate_schema()
        assert is_valid is True
        assert 'database' in schema
        assert 'collections' in schema
        assert 'document_structure' in schema
        print(f"   ✅ MongoDB schema validation successful")
        
        # Test disconnection
        disconnect_success = await connector.disconnect()
        assert disconnect_success is True
        print(f"   ✅ MongoDB disconnection successful")
    
    @pytest.mark.asyncio
    async def test_redis_connector(self):
        """Test Redis connector functionality"""
        print("\n🔴 Testing Redis Connector")
        
        config = AdvancedDataSourceConfig(
            source_type=DataDomain.GENERIC,
            connection_params={
                'host': 'localhost',
                'port': 6379,
                'database': 0
            },
            data_domain=DataDomain.GENERIC,
            security_level=DataSecurityLevel.INTERNAL
        )
        
        connector = RedisDataConnector(config)
        
        # Test connection
        success = await connector.connect()
        assert success is True
        print(f"   ✅ Redis connection successful")
        
        # Test data extraction
        data = await connector.extract()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'key' in data.columns
        assert 'value' in data.columns
        assert 'ttl' in data.columns
        print(f"   ✅ Redis data extraction: {len(data)} cache entries")
        
        # Test schema validation
        is_valid, schema = await connector.validate_schema()
        assert is_valid is True
        assert 'host' in schema
        assert 'port' in schema
        assert 'key_patterns' in schema
        print(f"   ✅ Redis schema validation successful")
        
        # Test disconnection
        disconnect_success = await connector.disconnect()
        assert disconnect_success is True
        print(f"   ✅ Redis disconnection successful")
    
    @pytest.mark.asyncio
    async def test_streaming_connector(self):
        """Test streaming data connector functionality"""
        print("\n📡 Testing Streaming Data Connector")
        
        config = AdvancedDataSourceConfig(
            source_type=DataDomain.IOT,
            connection_params={
                'stream_endpoint': 'ws://localhost:8080/stream',
                'buffer_size': 1000
            },
            data_domain=DataDomain.IOT,
            security_level=DataSecurityLevel.INTERNAL,
            refresh_interval=1
        )
        
        connector = StreamingDataConnector(config)
        
        # Test connection
        success = await connector.connect()
        assert success is True
        assert connector.is_streaming is True
        print(f"   ✅ Streaming connection successful")
        
        # Test data extraction
        data = await connector.extract()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert 'stream_id' in data.columns
        assert 'sensor_value' in data.columns
        assert 'timestamp' in data.columns
        print(f"   ✅ Streaming data extraction: {len(data)} sensor readings")
        
        # Test schema validation
        is_valid, schema = await connector.validate_schema()
        assert is_valid is True
        assert 'stream_type' in schema
        assert 'data_rate' in schema
        assert 'quality_metrics' in schema
        print(f"   ✅ Streaming schema validation successful")
        
        # Test disconnection
        disconnect_success = await connector.disconnect()
        assert disconnect_success is True
        assert connector.is_streaming is False
        print(f"   ✅ Streaming disconnection successful")


class TestEnhancedInsightGeneration:
    """Test enhanced insight generation capabilities"""
    
    @pytest.fixture
    def enhanced_generator(self):
        return EnhancedInsightGenerator()
    
    @pytest.mark.asyncio
    async def test_advanced_business_insights(self, enhanced_generator):
        """Test advanced business insight generation"""
        print("\n💡 Testing Advanced Business Insights")
        
        # Create comprehensive business data
        business_data = {
            'raw_data': {
                'financial': pd.DataFrame({
                    'revenue': np.random.exponential(10000, 100),
                    'profit': np.random.normal(5000, 1000, 100),
                    'sales_amount': np.random.exponential(1000, 100),
                    'cost': np.random.uniform(100, 5000, 100),
                    'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
                }),
                'customers': pd.DataFrame({
                    'customer_id': range(1, 101),
                    'customer_lifetime_value': np.random.exponential(5000, 100),
                    'churn_risk': np.random.uniform(0, 1, 100),
                    'user_satisfaction': np.random.uniform(1, 5, 100),
                    'retention_rate': np.random.uniform(0.7, 0.95, 100),
                    'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
                }),
                'operations': pd.DataFrame({
                    'efficiency_score': np.random.uniform(0.6, 0.95, 100),
                    'performance_metric': np.random.normal(85, 10, 100),
                    'uptime_percentage': np.random.uniform(0.95, 0.999, 100),
                    'response_time': np.random.exponential(200, 100),
                    'error_rate': np.random.uniform(0.001, 0.05, 100),
                    'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
                })
            }
        }
        
        insights = await enhanced_generator.generate_advanced_business_insights(business_data)
        
        print(f"   ✅ Advanced insights generated: {len(insights)}")
        
        # Check for different types of insights
        insight_types = set(insight['type'] for insight in insights)
        print(f"   ✅ Insight types: {list(insight_types)}")
        
        # Should generate various business insights
        assert len(insights) > 0
        
        # Check for specific business insight types
        revenue_insights = [i for i in insights if i['type'] == 'revenue_optimization']
        customer_insights = [i for i in insights if i['type'] == 'customer_lifecycle']
        efficiency_insights = [i for i in insights if i['type'] == 'operational_efficiency']
        risk_insights = [i for i in insights if i['type'] == 'risk_assessment']
        
        print(f"   ✅ Revenue insights: {len(revenue_insights)}")
        print(f"   ✅ Customer insights: {len(customer_insights)}")
        print(f"   ✅ Efficiency insights: {len(efficiency_insights)}")
        print(f"   ✅ Risk insights: {len(risk_insights)}")
        
        # Verify business impact calculations
        for insight in insights:
            if 'business_impact' in insight:
                assert insight['business_impact'] >= 0
                print(f"   💰 {insight['title']}: ${insight['business_impact']:,.0f} impact")
    
    @pytest.mark.asyncio
    async def test_revenue_optimization_insights(self, enhanced_generator):
        """Test revenue optimization insight generation"""
        print("\n💰 Testing Revenue Optimization Insights")
        
        # Create revenue data with clear trends
        revenue_data = {
            'revenue_source': pd.DataFrame({
                'revenue': np.concatenate([
                    np.random.normal(10000, 1000, 50),  # Stable period
                    np.random.normal(15000, 1500, 50)   # Growth period
                ]),
                'profit_margin': np.random.uniform(0.1, 0.3, 100),
                'sales_volume': np.random.poisson(100, 100),
                'price_point': np.random.uniform(50, 200, 100),
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
            })
        }
        
        revenue_insights = await enhanced_generator._generate_revenue_optimization_insights(revenue_data)
        
        print(f"   ✅ Revenue optimization insights: {len(revenue_insights)}")
        
        # Should detect the revenue trend
        if len(revenue_insights) > 0:
            for insight in revenue_insights:
                print(f"   📈 {insight['title']}: {insight['description']}")
                assert 'trend_percentage' in insight['metrics']
                assert 'business_impact' in insight
    
    @pytest.mark.asyncio
    async def test_risk_assessment_insights(self, enhanced_generator):
        """Test risk assessment insight generation"""
        print("\n⚠️ Testing Risk Assessment Insights")
        
        # Create data with risk indicators
        risk_data = {
            'risk_source': pd.DataFrame({
                'error_rate': np.concatenate([
                    np.random.uniform(0.001, 0.01, 80),  # Normal error rates
                    np.random.uniform(0.1, 0.5, 20)      # High error rates (risk)
                ]),
                'failure_count': np.concatenate([
                    np.random.poisson(2, 80),    # Normal failures
                    np.random.poisson(20, 20)    # High failures (risk)
                ]),
                'critical_alerts': np.random.randint(0, 10, 100),
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='h')
            })
        }
        
        risk_insights = await enhanced_generator._generate_risk_assessment_insights(risk_data)
        
        print(f"   ✅ Risk assessment insights: {len(risk_insights)}")
        
        # Should detect high risk indicators
        for insight in risk_insights:
            print(f"   🚨 {insight['title']}: {insight['description']}")
            assert 'risk_level' in insight['metrics']
            assert insight['importance_score'] > 0


class TestIntegratedEnhancedSystem:
    """Test the integrated enhanced system"""
    
    @pytest.fixture
    def enhanced_fusion_engine(self):
        engine = DataFusionEngine()
        # Replace with enhanced insight generator
        engine.insight_generator = EnhancedInsightGenerator()
        return engine
    
    @pytest.mark.asyncio
    async def test_multi_connector_integration(self, enhanced_fusion_engine):
        """Test integration of multiple enhanced connectors"""
        print("\n🔗 Testing Multi-Connector Integration")
        
        # Create CSV data
        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        csv_data = pd.DataFrame({
            'id': range(1, 101),
            'revenue': np.random.exponential(1000, 100),
            'customer_count': np.random.randint(10, 100, 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
        })
        csv_data.to_csv(csv_file.name, index=False)
        csv_file.close()
        
        # Register multiple connector types
        connectors = [
            ('csv_data', 'CSVDataConnector', DataDomain.FINANCIAL, {'file_path': csv_file.name}),
            ('mongo_data', 'MongoDBDataConnector', DataDomain.CUSTOMER, {'host': 'localhost', 'database': 'test'}),
            ('redis_cache', 'RedisDataConnector', DataDomain.OPERATIONAL, {'host': 'localhost', 'port': 6379}),
            ('stream_data', 'StreamingDataConnector', DataDomain.IOT, {'stream_endpoint': 'ws://localhost:8080'})
        ]
        
        registered_sources = []
        
        for name, connector_type, domain, params in connectors:
            config = AdvancedDataSourceConfig(
                source_type=domain,
                connection_params=params,
                data_domain=domain,
                security_level=DataSecurityLevel.INTERNAL
            )
            
            if connector_type == 'CSVDataConnector':
                connector = palantir_advanced_system.CSVDataConnector(config)
            elif connector_type == 'MongoDBDataConnector':
                connector = MongoDBDataConnector(config)
            elif connector_type == 'RedisDataConnector':
                connector = RedisDataConnector(config)
            elif connector_type == 'StreamingDataConnector':
                connector = StreamingDataConnector(config)
            
            success = await enhanced_fusion_engine.register_source(name, connector)
            if success:
                registered_sources.append(name)
                print(f"   ✅ {connector_type} registered as {name}")
        
        # Test federated query across multiple connector types
        if len(registered_sources) > 0:
            query = {
                'sources': {source: {'select': '*'} for source in registered_sources},
                'fusion_strategy': 'multi_layer'
            }
            
            results = await enhanced_fusion_engine.federated_query(query)
            
            print(f"   ✅ Multi-connector query completed")
            print(f"   ✅ Sources queried: {len(results['results'])}")
            print(f"   ✅ Total data points: {results['stats']['data_points_processed']}")
            print(f"   ✅ Enhanced insights: {len(results['insights'])}")
            
            assert len(results['results']) == len(registered_sources)
            assert results['stats']['data_points_processed'] > 0
            assert len(results['insights']) > 0
        
        # Clean up
        os.unlink(csv_file.name)
    
    @pytest.mark.asyncio
    async def test_enhanced_business_scenario(self, enhanced_fusion_engine):
        """Test enhanced business scenario with advanced insights"""
        print("\n🏢 Testing Enhanced Business Scenario")
        
        # Create comprehensive business dataset
        business_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Generate realistic business data with trends and patterns
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        base_revenue = 50000
        growth_rate = 0.02  # 2% growth
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * np.arange(365) / 365)
        
        business_data = pd.DataFrame({
            'date': dates,
            'revenue': base_revenue * (1 + growth_rate) ** np.arange(365) * seasonal_factor + np.random.normal(0, 2000, 365),
            'profit': np.random.normal(15000, 3000, 365),
            'sales_volume': np.random.poisson(500, 365),
            'customer_acquisition_cost': np.random.uniform(50, 200, 365),
            'customer_lifetime_value': np.random.exponential(2000, 365),
            'churn_rate': np.random.uniform(0.02, 0.08, 365),
            'efficiency_score': np.random.uniform(0.7, 0.95, 365),
            'error_rate': np.random.uniform(0.001, 0.02, 365),
            'risk_score': np.random.uniform(0.1, 0.8, 365)
        })
        
        business_data.to_csv(business_file.name, index=False)
        business_file.close()
        
        # Register business data
        config = AdvancedDataSourceConfig(
            source_type=DataDomain.FINANCIAL,
            connection_params={'file_path': business_file.name},
            data_domain=DataDomain.FINANCIAL,
            security_level=DataSecurityLevel.CONFIDENTIAL
        )
        
        connector = palantir_advanced_system.CSVDataConnector(config)
        await enhanced_fusion_engine.register_source('business_data', connector)
        
        # Execute enhanced business analysis
        query = {
            'sources': {
                'business_data': {'select': '*'}
            },
            'fusion_strategy': 'multi_layer',
            'metadata': {
                'analysis_type': 'comprehensive_business_intelligence',
                'include_predictions': True,
                'risk_assessment': True
            }
        }
        
        start_time = time.time()
        results = await enhanced_fusion_engine.federated_query(query)
        execution_time = time.time() - start_time
        
        print(f"   ✅ Enhanced business analysis time: {execution_time:.3f}s")
        print(f"   ✅ Business data points: {results['stats']['data_points_processed']}")
        print(f"   ✅ Enhanced insights: {len(results['insights'])}")
        
        # Analyze insight types
        insight_types = {}
        for insight in results['insights']:
            insight_type = insight['type']
            if insight_type not in insight_types:
                insight_types[insight_type] = 0
            insight_types[insight_type] += 1
        
        print(f"   ✅ Insight breakdown:")
        for insight_type, count in insight_types.items():
            print(f"      - {insight_type}: {count}")
        
        # Should generate comprehensive business insights
        assert results['stats']['data_points_processed'] == 365
        assert len(results['insights']) > 0
        assert execution_time < 5.0
        
        # Check for business impact calculations
        business_impact_insights = [i for i in results['insights'] if 'business_impact' in i]
        total_business_impact = sum(i['business_impact'] for i in business_impact_insights)
        
        print(f"   💰 Total business impact: ${total_business_impact:,.0f}")
        
        # Clean up
        os.unlink(business_file.name)


class TestProductionDeploymentFeatures:
    """Test production deployment features"""
    
    @pytest.fixture
    def production_engine(self):
        return DataFusionEngine()
    
    @pytest.mark.asyncio
    async def test_health_check_system(self, production_engine):
        """Test system health check capabilities"""
        print("\n🏥 Testing Health Check System")
        
        # Test engine health
        assert hasattr(production_engine, 'connections')
        assert hasattr(production_engine, 'schema_registry')
        print(f"   ✅ Engine components healthy")
        
        # Test connector health
        config = AdvancedDataSourceConfig(
            source_type=DataDomain.OPERATIONAL,
            connection_params={'host': 'localhost'},
            data_domain=DataDomain.OPERATIONAL,
            security_level=DataSecurityLevel.INTERNAL
        )
        
        connector = palantir_advanced_system.SQLDataConnector(config)
        health_check = await connector.connect()
        
        assert health_check is True
        assert connector.status.value == "connected"
        print(f"   ✅ Connector health check passed")
        
        # Test performance metrics
        assert hasattr(connector, 'performance_metrics')
        assert hasattr(connector.performance_metrics, 'latency')
        assert hasattr(connector.performance_metrics, 'throughput')
        assert hasattr(connector.performance_metrics, 'error_rate')
        print(f"   ✅ Performance metrics available")
    
    def test_configuration_management(self):
        """Test advanced configuration management"""
        print("\n⚙️ Testing Configuration Management")
        
        # Test configuration validation
        configs = [
            # Minimal configuration
            AdvancedDataSourceConfig(
                source_type=DataDomain.GENERIC,
                connection_params={'host': 'localhost'},
                data_domain=DataDomain.GENERIC,
                security_level=DataSecurityLevel.PUBLIC
            ),
            # Full configuration
            AdvancedDataSourceConfig(
                source_type=DataDomain.FINANCIAL,
                connection_params={'host': 'localhost', 'port': 5432, 'database': 'finance'},
                data_domain=DataDomain.FINANCIAL,
                security_level=DataSecurityLevel.CONFIDENTIAL,
                max_workers=16,
                cache_enabled=True,
                debug=True,
                sampling_rate=0.8,
                max_retries=5,
                timeout=60,
                metadata={'environment': 'production', 'version': '1.0'}
            )
        ]
        
        for i, config in enumerate(configs):
            print(f"   ✅ Configuration {i+1} validated")
            assert config.source_type in DataDomain
            assert config.security_level in DataSecurityLevel
            assert 1 <= config.refresh_interval <= 86400
            assert 1 <= config.priority <= 5
            assert config.max_retries >= 0
            assert config.timeout > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_and_alerting(self, production_engine):
        """Test monitoring and alerting capabilities"""
        print("\n📊 Testing Monitoring and Alerting")
        
        # Create data source for monitoring
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        monitoring_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='min'),
            'cpu_usage': np.random.uniform(0.1, 0.9, 100),
            'memory_usage': np.random.uniform(0.2, 0.95, 100),
            'disk_usage': np.random.uniform(0.1, 0.8, 100),
            'network_latency': np.random.exponential(50, 100),
            'error_count': np.random.poisson(5, 100),
            'active_users': np.random.randint(100, 1000, 100)
        })
        monitoring_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        config = AdvancedDataSourceConfig(
            source_type=DataDomain.OPERATIONAL,
            connection_params={'file_path': temp_file.name},
            data_domain=DataDomain.OPERATIONAL,
            security_level=DataSecurityLevel.INTERNAL,
            refresh_interval=60  # Monitor every minute
        )
        
        connector = palantir_advanced_system.CSVDataConnector(config)
        await production_engine.register_source('monitoring', connector)
        
        # Execute monitoring query
        query = {
            'sources': {
                'monitoring': {'select': '*'}
            },
            'fusion_strategy': 'multi_layer',
            'metadata': {
                'monitoring_mode': True,
                'alert_thresholds': {
                    'cpu_usage': 0.8,
                    'memory_usage': 0.9,
                    'error_count': 10
                }
            }
        }
        
        results = await production_engine.federated_query(query)
        
        print(f"   ✅ Monitoring data points: {results['stats']['data_points_processed']}")
        print(f"   ✅ Monitoring insights: {len(results['insights'])}")
        
        # Should detect operational issues
        operational_insights = [i for i in results['insights'] 
                              if i['type'] in ['anomaly', 'operational_efficiency', 'risk_assessment']]
        
        print(f"   ✅ Operational insights: {len(operational_insights)}")
        
        assert results['stats']['data_points_processed'] == 100
        assert len(results['insights']) > 0
        
        # Clean up
        os.unlink(temp_file.name)


if __name__ == "__main__":
    # Run the enhanced connector tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
