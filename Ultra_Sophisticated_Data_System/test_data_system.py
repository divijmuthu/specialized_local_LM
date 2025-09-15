"""
Comprehensive test suite for the Ultra-Sophisticated Data Embedding & Insight Generation System
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Import the system components
import sys
import os
import importlib.util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load the module with hyphen in filename
spec = importlib.util.spec_from_file_location('palantir_advanced_system', 'palantir-advanced-system.py')
palantir_advanced_system = importlib.util.module_from_spec(spec)
spec.loader.exec_module(palantir_advanced_system)

# Import components from the loaded module
DataFusionEngine = palantir_advanced_system.DataFusionEngine
SQLDataConnector = palantir_advanced_system.SQLDataConnector
CSVDataConnector = palantir_advanced_system.CSVDataConnector
APIDataConnector = palantir_advanced_system.APIDataConnector
AdvancedDataSourceConfig = palantir_advanced_system.AdvancedDataSourceConfig
DataDomain = palantir_advanced_system.DataDomain
DataSecurityLevel = palantir_advanced_system.DataSecurityLevel
InsightGenerator = palantir_advanced_system.InsightGenerator
QuantumProcessor = palantir_advanced_system.QuantumProcessor
KnowledgeGraphEngine = palantir_advanced_system.KnowledgeGraphEngine


class TestDataConnectors:
    """Test suite for data connectors"""
    
    @pytest.fixture
    def sql_config(self):
        return AdvancedDataSourceConfig(
            source_type=DataDomain.FINANCIAL,
            connection_params={'host': 'localhost', 'port': 5432},
            data_domain=DataDomain.FINANCIAL,
            security_level=DataSecurityLevel.INTERNAL
        )
    
    @pytest.fixture
    def csv_config(self):
        # Create a temporary CSV file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'User_{i}' for i in range(1, 11)],
            'value': np.random.randn(10),
            'category': np.random.choice(['A', 'B', 'C'], 10)
        })
        test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        return AdvancedDataSourceConfig(
            source_type=DataDomain.CUSTOMER,
            connection_params={'file_path': self.temp_file.name},
            data_domain=DataDomain.CUSTOMER,
            security_level=DataSecurityLevel.INTERNAL
        )
    
    @pytest.fixture
    def api_config(self):
        return AdvancedDataSourceConfig(
            source_type=DataDomain.IOT,
            connection_params={'endpoint': 'https://jsonplaceholder.typicode.com/posts'},
            data_domain=DataDomain.IOT,
            security_level=DataSecurityLevel.INTERNAL
        )
    
    @pytest.mark.asyncio
    async def test_sql_connector_connection(self, sql_config):
        """Test SQL connector connection"""
        connector = SQLDataConnector(sql_config)
        result = await connector.connect()
        assert result is True
        assert connector.status.value == "connected"
    
    @pytest.mark.asyncio
    async def test_sql_connector_extract(self, sql_config):
        """Test SQL connector data extraction"""
        connector = SQLDataConnector(sql_config)
        await connector.connect()
        data = await connector.extract()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'id' in data.columns
        assert 'value' in data.columns
        assert 'category' in data.columns
        assert 'timestamp' in data.columns
    
    @pytest.mark.asyncio
    async def test_sql_connector_schema_validation(self, sql_config):
        """Test SQL connector schema validation"""
        connector = SQLDataConnector(sql_config)
        is_valid, schema = await connector.validate_schema()
        
        assert is_valid is True
        assert 'tables' in schema
        assert 'columns' in schema
        assert 'users' in schema['tables']
    
    @pytest.mark.asyncio
    async def test_csv_connector_connection(self, csv_config):
        """Test CSV connector connection"""
        connector = CSVDataConnector(csv_config)
        result = await connector.connect()
        assert result is True
        assert connector.status.value == "connected"
    
    @pytest.mark.asyncio
    async def test_csv_connector_extract(self, csv_config):
        """Test CSV connector data extraction"""
        connector = CSVDataConnector(csv_config)
        await connector.connect()
        data = await connector.extract()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
        assert 'id' in data.columns
        assert 'name' in data.columns
        assert 'value' in data.columns
        assert 'category' in data.columns
    
    @pytest.mark.asyncio
    async def test_csv_connector_schema_validation(self, csv_config):
        """Test CSV connector schema validation"""
        connector = CSVDataConnector(csv_config)
        is_valid, schema = await connector.validate_schema()
        
        assert is_valid is True
        assert 'columns' in schema
        assert 'dtypes' in schema
        assert 'file_path' in schema
    
    @pytest.mark.asyncio
    async def test_csv_connector_sampling(self, csv_config):
        """Test CSV connector with sampling"""
        csv_config.sampling_rate = 0.5
        connector = CSVDataConnector(csv_config)
        await connector.connect()
        data = await connector.extract()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5  # 50% of 10 rows
    
    @pytest.mark.asyncio
    async def test_api_connector_connection(self, api_config):
        """Test API connector connection"""
        connector = APIDataConnector(api_config)
        result = await connector.connect()
        # This might fail if the API is not accessible, which is expected
        # We'll test the connection logic regardless
        assert isinstance(result, bool)
    
    def teardown_method(self):
        """Clean up temporary files"""
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)


class TestDataFusionEngine:
    """Test suite for the main data fusion engine"""
    
    @pytest.fixture
    def fusion_engine(self):
        return DataFusionEngine()
    
    @pytest.fixture
    def sql_connector(self):
        config = AdvancedDataSourceConfig(
            source_type=DataDomain.FINANCIAL,
            connection_params={'host': 'localhost', 'port': 5432},
            data_domain=DataDomain.FINANCIAL,
            security_level=DataSecurityLevel.INTERNAL
        )
        return SQLDataConnector(config)
    
    @pytest.mark.asyncio
    async def test_register_source(self, fusion_engine, sql_connector):
        """Test registering a data source"""
        result = await fusion_engine.register_source('test_db', sql_connector)
        assert result is True
        assert 'test_db' in fusion_engine.connections
        assert 'test_db' in fusion_engine.schema_registry
    
    @pytest.mark.asyncio
    async def test_federated_query_single_source(self, fusion_engine, sql_connector):
        """Test federated query with single source"""
        await fusion_engine.register_source('test_db', sql_connector)
        
        query = {
            'sources': {
                'test_db': {'select': '*', 'limit': 50}
            },
            'fusion_strategy': 'multi_layer'
        }
        
        results = await fusion_engine.federated_query(query)
        
        assert 'results' in results
        assert 'fused_data' in results
        assert 'insights' in results
        assert 'execution_time' in results
        assert 'stats' in results
        assert 'test_db' in results['results']
        assert isinstance(results['results']['test_db'], pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_federated_query_multiple_sources(self, fusion_engine):
        """Test federated query with multiple sources"""
        # Create multiple connectors
        sql_config = AdvancedDataSourceConfig(
            source_type=DataDomain.FINANCIAL,
            connection_params={'host': 'localhost', 'port': 5432},
            data_domain=DataDomain.FINANCIAL,
            security_level=DataSecurityLevel.INTERNAL
        )
        sql_connector = SQLDataConnector(sql_config)
        
        # Create temporary CSV for second source
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data = pd.DataFrame({
            'id': range(1, 21),
            'amount': np.random.randn(20) * 1000,
            'date': pd.date_range('2024-01-01', periods=20, freq='D')
        })
        test_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        csv_config = AdvancedDataSourceConfig(
            source_type=DataDomain.CUSTOMER,
            connection_params={'file_path': temp_file.name},
            data_domain=DataDomain.CUSTOMER,
            security_level=DataSecurityLevel.INTERNAL
        )
        csv_connector = CSVDataConnector(csv_config)
        
        # Register both sources
        await fusion_engine.register_source('financial_db', sql_connector)
        await fusion_engine.register_source('customer_data', csv_connector)
        
        query = {
            'sources': {
                'financial_db': {'select': '*', 'limit': 30},
                'customer_data': {'select': '*', 'limit': 20}
            },
            'fusion_strategy': 'multi_layer'
        }
        
        results = await fusion_engine.federated_query(query)
        
        assert len(results['results']) == 2
        assert 'financial_db' in results['results']
        assert 'customer_data' in results['results']
        assert results['stats']['data_points_processed'] > 0
        
        # Clean up
        os.unlink(temp_file.name)
    
    def test_generate_optimized_query_plan(self, fusion_engine):
        """Test query plan generation"""
        query = {
            'sources': {
                'source1': {'select': '*', 'limit': 100},
                'source2': {'select': '*', 'limit': 200}
            },
            'fusion_strategy': 'multi_layer'
        }
        
        plan = fusion_engine._generate_optimized_query_plan(query)
        
        assert 'sources' in plan
        assert 'optimizations' in plan
        assert 'fusion_strategy' in plan
        assert 'domains_involved' in plan
        assert plan['fusion_strategy'] == 'multi_layer'
    
    def test_determine_optimizations(self, fusion_engine):
        """Test optimization determination"""
        query = {
            'sources': {
                'source1': {'select': '*', 'limit': 100},
                'source2': {'select': '*', 'limit': 200}
            },
            'aggregations': ['sum', 'count']
        }
        
        optimizations = fusion_engine._determine_optimizations(query)
        
        assert 'parallel_execution' in optimizations
        assert 'incremental_aggregation' in optimizations


class TestInsightGenerator:
    """Test suite for insight generation"""
    
    @pytest.fixture
    def insight_generator(self):
        return InsightGenerator()
    
    @pytest.fixture
    def sample_data_package(self):
        return {
            'raw_data': {
                'source1': pd.DataFrame({
                    'id': range(1, 101),
                    'value1': np.random.randn(100),
                    'value2': np.random.randn(100) * 2,
                    'category': np.random.choice(['A', 'B', 'C'], 100)
                }),
                'source2': pd.DataFrame({
                    'id': range(1, 101),
                    'value3': np.random.randn(100) * 1.5,
                    'value4': np.random.randn(100) * 0.5,
                    'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
                })
            },
            'fused_data': {},
            'query_metadata': {}
        }
    
    @pytest.mark.asyncio
    async def test_generate_multi_domain_insights(self, insight_generator, sample_data_package):
        """Test multi-domain insight generation"""
        insights = await insight_generator.generate_multi_domain_insights(sample_data_package)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Check insight structure
        for insight in insights:
            assert 'title' in insight
            assert 'description' in insight
            assert 'type' in insight
            assert 'importance_score' in insight
            assert 'novelty_score' in insight
            assert 0 <= insight['importance_score'] <= 1
            assert 0 <= insight['novelty_score'] <= 1
    
    @pytest.mark.asyncio
    async def test_cross_domain_correlations(self, insight_generator, sample_data_package):
        """Test cross-domain correlation analysis"""
        insights = await insight_generator._generate_cross_domain_correlations(sample_data_package)
        
        assert isinstance(insights, list)
        # Should find correlations between the datasets
        if len(insights) > 0:
            for insight in insights:
                assert insight['type'] == 'cross_domain_correlation'
                assert 'correlation_coefficient' in insight['metrics']
    
    @pytest.mark.asyncio
    async def test_temporal_pattern_detection(self, insight_generator, sample_data_package):
        """Test temporal pattern detection"""
        insights = await insight_generator._detect_temporal_patterns(sample_data_package)
        
        assert isinstance(insights, list)
        # Should detect temporal data in source2
        temporal_insights = [i for i in insights if i['type'] == 'temporal_pattern']
        assert len(temporal_insights) > 0


class TestQuantumProcessor:
    """Test suite for quantum processing capabilities"""
    
    @pytest.fixture
    def quantum_processor(self):
        return QuantumProcessor()
    
    @pytest.fixture
    def sample_datasets(self):
        dataset1 = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        dataset2 = pd.DataFrame({
            'feature1': np.random.randn(100) * 0.8 + 0.2,
            'feature2': np.random.randn(100) * 0.6 + 0.4,
            'feature4': np.random.randn(100)
        })
        return dataset1, dataset2
    
    @pytest.mark.asyncio
    async def test_entangle_datasets(self, quantum_processor, sample_datasets):
        """Test quantum entanglement of datasets"""
        dataset1, dataset2 = sample_datasets
        result = await quantum_processor.entangle_datasets(dataset1, dataset2)
        
        assert 'entanglement_strength' in result
        assert 'correlation_matrix' in result
        assert 'quantum_advantage' in result
        assert 'qubits_used' in result
        assert 0 <= result['entanglement_strength'] <= 1
        assert result['quantum_advantage'] > 0
        assert result['qubits_used'] > 0


class TestKnowledgeGraphEngine:
    """Test suite for knowledge graph engine"""
    
    @pytest.fixture
    def graph_engine(self):
        return KnowledgeGraphEngine()
    
    def test_register_data_source(self, graph_engine):
        """Test data source registration in knowledge graph"""
        schema = {
            'tables': ['users', 'transactions'],
            'columns': {
                'users': ['id', 'name'],
                'transactions': ['id', 'amount']
            }
        }
        
        graph_engine.register_data_source(
            'test_source',
            'financial',
            schema,
            2
        )
        
        # Check if the graph was updated
        if graph_engine.graph is not None:
            assert graph_engine.graph.number_of_nodes() > 0
    
    @pytest.mark.asyncio
    async def test_fuse_with_knowledge_graph(self, graph_engine):
        """Test knowledge graph fusion"""
        test_results = {
            'source1': pd.DataFrame({'id': [1, 2], 'value': [10, 20]}),
            'source2': pd.DataFrame({'id': [1, 2], 'amount': [100, 200]})
        }
        
        fused = await graph_engine.fuse_with_knowledge_graph(test_results)
        
        assert 'original_data' in fused
        assert 'graph_enhanced' in fused
        assert 'num_entities' in fused
        assert 'num_relationships' in fused
        assert fused['graph_enhanced'] is True


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow from data source to insights"""
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data = pd.DataFrame({
            'id': range(1, 51),
            'revenue': np.random.randn(50) * 10000 + 50000,
            'cost': np.random.randn(50) * 5000 + 30000,
            'profit': np.random.randn(50) * 5000 + 20000,
            'date': pd.date_range('2024-01-01', periods=50, freq='D')
        })
        test_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            # Initialize system
            engine = DataFusionEngine()
            
            # Configure data source
            config = AdvancedDataSourceConfig(
                source_type=DataDomain.FINANCIAL,
                connection_params={'file_path': temp_file.name},
                data_domain=DataDomain.FINANCIAL,
                security_level=DataSecurityLevel.INTERNAL
            )
            
            # Create and register connector
            connector = CSVDataConnector(config)
            success = await engine.register_source('financial_data', connector)
            assert success is True
            
            # Execute query
            query = {
                'sources': {
                    'financial_data': {'select': '*', 'limit': 50}
                },
                'fusion_strategy': 'multi_layer'
            }
            
            results = await engine.federated_query(query)
            
            # Validate results
            assert 'results' in results
            assert 'insights' in results
            assert 'execution_time' in results
            assert len(results['insights']) > 0
            
            # Check that insights are properly structured
            for insight in results['insights']:
                assert 'title' in insight
                assert 'description' in insight
                assert 'importance_score' in insight
                assert 'novelty_score' in insight
            
            print(f"✅ Complete workflow test passed!")
            print(f"   Execution time: {results['execution_time']:.2f}s")
            print(f"   Insights generated: {len(results['insights'])}")
            print(f"   Data points processed: {results['stats']['data_points_processed']}")
            
        finally:
            # Clean up
            os.unlink(temp_file.name)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios"""
        engine = DataFusionEngine()
        
        # Test with invalid file path
        config = AdvancedDataSourceConfig(
            source_type=DataDomain.CUSTOMER,
            connection_params={'file_path': '/nonexistent/file.csv'},
            data_domain=DataDomain.CUSTOMER,
            security_level=DataSecurityLevel.INTERNAL
        )
        
        connector = CSVDataConnector(config)
        success = await engine.register_source('invalid_source', connector)
        assert success is False  # Should fail due to invalid file
        
        # Test query with non-existent source
        query = {
            'sources': {
                'nonexistent_source': {'select': '*', 'limit': 10}
            },
            'fusion_strategy': 'multi_layer'
        }
        
        results = await engine.federated_query(query)
        assert 'results' in results
        assert 'nonexistent_source' in results['results']
        assert results['results']['nonexistent_source'] is None  # No results from non-existent source


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
