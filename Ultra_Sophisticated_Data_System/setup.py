"""
Setup script for Ultra-Sophisticated Data Embedding & Insight Generation System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Ultra-Sophisticated Data Embedding & Insight Generation System"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="ultra-sophisticated-data-system",
    version="1.0.0",
    author="Data Systems Team",
    author_email="data-systems@example.com",
    description="Ultra-Sophisticated Data Embedding & Insight Generation System that surpasses Palantir",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/ultra-sophisticated-data-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pydantic>=1.8.0",
        "aiohttp>=3.8.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "standard": [
            "scikit-learn>=1.0.0",
            "networkx>=2.6.0",
            "matplotlib>=3.5.0",
            "sqlalchemy>=1.4.0",
            "psycopg2-binary>=2.9.0",
        ],
        "full": [
            "tensorflow>=2.8.0",
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "catboost>=1.1.0",
            "spacy>=3.4.0",
            "transformers>=4.20.0",
            "sentence-transformers>=2.2.0",
            "statsmodels>=0.13.0",
            "ray>=2.0.0",
            "dask>=2022.5.0",
            "faiss-cpu>=1.7.0",
            "pymongo>=4.0.0",
            "redis>=4.3.0",
            "cassandra-driver>=3.25.0",
            "snowflake-connector-python>=2.7.0",
            "boto3>=1.24.0",
            "azure-storage-blob>=12.12.0",
            "google-cloud-bigquery>=3.2.0",
            "pyarrow>=8.0.0",
            "seaborn>=0.11.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.19.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "data-system=palantir_advanced_system:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="data-analysis, machine-learning, data-fusion, insights, palantir-alternative",
    project_urls={
        "Bug Reports": "https://github.com/example/ultra-sophisticated-data-system/issues",
        "Source": "https://github.com/example/ultra-sophisticated-data-system",
        "Documentation": "https://github.com/example/ultra-sophisticated-data-system/wiki",
    },
)
