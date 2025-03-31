"""
Experiments module for RAG testing framework.

This module provides utilities for running experiments with different RAG configurations.
"""

from rag_testing.experiments.experiment_runner import (
    run_retrieval_experiment,
    evaluate_search_results,
    get_model_details,
    get_tokenizer_details,
    get_tfidf_details
)
from rag_testing.experiments.data import (
    create_test_documents,
    create_test_query_relevance,
    get_standard_test_queries
)
from rag_testing.experiments.configs import (
    get_experiment_config,
    create_embedding_config,
    create_hybrid_config,
    create_tfidf_config
)
from rag_testing.experiments.runner import run_experiment

__all__ = [
    # Runner
    "run_retrieval_experiment",
    "run_experiment",
    
    # Evaluation
    "evaluate_search_results",
    
    # Configurations
    "get_experiment_config",
    "create_embedding_config",
    "create_hybrid_config",
    "create_tfidf_config",
    
    # Data
    "create_test_documents",
    "create_test_query_relevance",
    "get_standard_test_queries",
    
    # Utilities
    "get_model_details",
    "get_tokenizer_details",
    "get_tfidf_details"
] 