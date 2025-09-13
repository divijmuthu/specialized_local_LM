"""
Pipeline module for data collection, ETL, and processing
"""
from .data_collectors import (
    DataCollector,
    CRMDataCollector,
    SalesDataCollector,
    FinancialDataCollector,
    ExternalDataCollector,
    DataCollectionOrchestrator,
    SampleDataGenerator
)
from .etl_pipeline import ETLPipeline

__all__ = [
    'DataCollector',
    'CRMDataCollector',
    'SalesDataCollector',
    'FinancialDataCollector',
    'ExternalDataCollector',
    'DataCollectionOrchestrator',
    'SampleDataGenerator',
    'ETLPipeline'
]
