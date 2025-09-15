"""
Ultra-Sophisticated Data Embedding & Insight Generation System
Simple example script matching the specification
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.util

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


async def start_system():
    """Start the main system as shown in the specification"""
    print("🚀 Initializing Ultra-Sophisticated Data Embedding & Insight Generation System...")
    
    # Create the main engine
    engine = DataFusionEngine()
    
    # Configure a data source
    config = AdvancedDataSourceConfig(
        source_type=DataDomain.FINANCIAL,
        connection_params={'host': 'localhost', 'port': 5432},
        data_domain=DataDomain.FINANCIAL,
        security_level=DataSecurityLevel.INTERNAL
    )
    
    # Create connector
    connector = SQLDataConnector(config)
    
    # Register the source
    success = await engine.register_source('my_database', connector)
    if success:
        print("✅ System ready!")
        return engine
    else:
        print("❌ System initialization failed!")
        return None


async def query_data(engine):
    """Query data as shown in the specification"""
    query = {
        'sources': {
            'my_database': {'select': '*', 'limit': 100}
        },
        'fusion_strategy': 'multi_layer'
    }
    results = await engine.federated_query(query)
    return results


async def demonstrate_system():
    """Demonstrate the complete system functionality"""
    print("=" * 60)
    print("Ultra-Sophisticated Data Embedding & Insight Generation System")
    print("=" * 60)
    
    # Start the system
    engine = await start_system()
    if not engine:
        return
    
    print("\n📊 Executing federated query...")
    
    # Run query
    results = await query_data(engine)
    
    print("\n" + "=" * 60)
    print("QUERY RESULTS")
    print("=" * 60)
    
    # Display results as shown in the specification
    print(f"Execution time: {results['execution_time']:.2f} seconds")
    print(f"Data points processed: {results['stats']['data_points_processed']}")
    print(f"Fusion strategy: {results['stats']['fusion_strategy']}")
    print(f"Domains involved: {results['stats']['domains_involved']}")
    
    print(f"\n📈 Generated {len(results['insights'])} insights:")
    for i, insight in enumerate(results['insights'][:5], 1):
        print(f"\n{i}. {insight['title']}")
        print(f"   {insight['description']}")
        print(f"   Type: {insight['type']}")
        print(f"   Importance: {insight['importance_score']:.2f}")
        print(f"   Novelty: {insight['novelty_score']:.2f}")
    
    if len(results['insights']) > 5:
        print(f"\n... and {len(results['insights']) - 5} more insights")
    
    print("\n" + "=" * 60)
    print("SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 60)


async def test_installation():
    """Test installation as shown in the specification"""
    print("Testing installation...")
    
    try:
        import pandas
        print("✓ Pandas installed")
    except ImportError:
        print("✗ Pandas missing")
    
    try:
        import numpy
        print("✓ NumPy installed")
    except ImportError:
        print("✗ NumPy missing")
    
    try:
        import tensorflow
        print("✓ TensorFlow installed (Advanced features available)")
    except ImportError:
        print("✗ TensorFlow missing (Basic features only)")
    
    try:
        import asyncio
        print("✓ asyncio available")
    except ImportError:
        print("✗ asyncio missing")
    
    try:
        import pydantic
        print("✓ Pydantic installed")
    except ImportError:
        print("✗ Pydantic missing")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Sophisticated Data System")
    parser.add_argument("--test", action="store_true", help="Test installation")
    parser.add_argument("--demo", action="store_true", help="Run system demonstration")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_installation())
    elif args.demo:
        asyncio.run(demonstrate_system())
    else:
        # Default: run the main demonstration
        asyncio.run(demonstrate_system())
