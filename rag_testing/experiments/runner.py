#!/usr/bin/env python
"""
Experiment Runner

This module provides a command-line interface for running RAG experiments
with different pipeline configurations and evaluation metrics.
"""

import os
import argparse
import time
from typing import Dict, Any, List, Optional, Callable
import json
import csv
import mlflow
from datetime import datetime

from rag_testing.core.pipeline import create_pipeline_from_config
from rag_testing.experiments.configs import get_experiment_config, create_tfidf_config
from rag_testing.experiments.data import (
    create_test_documents,
    create_test_query_relevance,
    get_standard_test_queries
)
from rag_testing.experiments.experiment_runner import (
    run_retrieval_experiment,
    get_model_details,
    get_tokenizer_details,
    get_tfidf_details
)

def run_experiment(
    experiment_type: str,
    model_name: str = "all-MiniLM-L6-v2",
    embedding_weight: float = 0.7,
    tfidf_weight: float = 0.3,
    top_k: int = 5,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    experiment_name: Optional[str] = None,
    custom_queries: Optional[List[str]] = None,
    evaluate_metrics: bool = True
) -> Dict[str, Any]:
    """
    Run an experiment with specified configuration.
    
    Args:
        experiment_type: Type of experiment ('embedding', 'hybrid', or 'tfidf')
        model_name: Name of the embedding model
        embedding_weight: Weight for embedding scores (for hybrid retrieval)
        tfidf_weight: Weight for TF-IDF scores (for hybrid retrieval)
        top_k: Number of results to retrieve
        chunk_size: Size of document chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        experiment_name: Custom name for the experiment
        custom_queries: Custom queries to use (default: standard test queries)
        evaluate_metrics: Whether to evaluate metrics
        
    Returns:
        Experiment results dictionary
    """
    print(f"Running {experiment_type} experiment")
    
    # Create configuration based on experiment type
    config_kwargs = {"top_k": top_k}
    
    # Only add model_name for embedding and hybrid experiments
    if experiment_type in ["embedding", "hybrid"]:
        config_kwargs["model_name"] = model_name
    
    # Only add weights for hybrid experiments
    if experiment_type == "hybrid":
        config_kwargs.update({
            "embedding_weight": embedding_weight,
            "tfidf_weight": tfidf_weight
        })
        print(f"Embedding weight: {embedding_weight}, TF-IDF weight: {tfidf_weight}")
    
    config = get_experiment_config(experiment_type, **config_kwargs)
    
    # Create pipeline
    pipeline = create_pipeline_from_config(config)
    
    # Set up metadata for experiment
    if experiment_name is None:
        experiment_name = f"{experiment_type}_experiments"
    
    # Prepare queries
    test_queries = custom_queries if custom_queries else get_standard_test_queries()
    
    # Get model details if applicable
    config_params = {
        "experiment_type": experiment_type,
        "top_k": top_k
    }
    
    if experiment_type in ["embedding", "hybrid"]:
        model_details = get_model_details(model_name)
        for key, value in model_details.items():
            if isinstance(value, (str, int, float, bool)):
                config_params[f"model_{key}"] = value
    
    if experiment_type in ["hybrid", "tfidf"]:
        tfidf_details = get_tfidf_details()
        for key, value in tfidf_details.items():
            if isinstance(value, (str, int, float, bool)):
                config_params[f"tfidf_{key}"] = value
    
    if experiment_type == "hybrid":
        config_params.update({
            "embedding_weight": embedding_weight,
            "tfidf_weight": tfidf_weight
        })
    
    # Set up result formatter for hybrid retrieval
    result_formatter = None
    if experiment_type == "hybrid":
        from rag_testing.experiments.formatters import format_hybrid_result
        result_formatter = format_hybrid_result
    
    # Run the experiment
    return run_retrieval_experiment(
        pipeline=pipeline,
        config_params=config_params,
        test_queries=test_queries,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        experiment_name=experiment_name,
        result_formatter=result_formatter,
        evaluate_metrics=evaluate_metrics
    )

def main():
    """Run experiments from command line."""
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["embedding", "hybrid", "tfidf"],
        default="embedding",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model"
    )
    parser.add_argument(
        "--embedding-weight", 
        type=float, 
        default=0.7,
        help="Weight for embedding scores (for hybrid retrieval)"
    )
    parser.add_argument(
        "--tfidf-weight", 
        type=float, 
        default=0.3,
        help="Weight for TF-IDF scores (for hybrid retrieval)"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5,
        help="Number of results to retrieve"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=200,
        help="Size of document chunks in tokens"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=50,
        help="Overlap between chunks in tokens"
    )

    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default=None,
        help="Custom name for the experiment"
    )
    parser.add_argument(
        "--no-metrics", 
        action="store_true",
        help="Disable evaluation metrics"
    )
    parser.add_argument(
        "--weight-comparison", 
        action="store_true",
        help="Run weight comparison for hybrid retrieval"
    )
    
    args = parser.parse_args()
    
    # Run weight comparison if requested
    if args.weight_comparison and args.type == "hybrid":
        print("Running hybrid weight comparison experiment")
        
        # Define weight combinations to test
        weight_combinations = [
            (1.0, 0.0),    # Pure embedding
            (0.8, 0.2),    # Embedding-heavy
            (0.7, 0.3),    # Embedding-heavy balanced
            (0.5, 0.5),    # Equal weights
            (0.3, 0.7),    # TF-IDF-heavy balanced
            (0.2, 0.8),    # TF-IDF-heavy
            (0.0, 1.0)     # Pure TF-IDF
        ]
        
        # Run experiments with different weights
        results = {}
        for embedding_weight, tfidf_weight in weight_combinations:
            print(f"\nTesting weights: Embedding = {embedding_weight}, TF-IDF = {tfidf_weight}")
            
            exp_result = run_experiment(
                experiment_type="hybrid",
                model_name=args.model,
                embedding_weight=embedding_weight,
                tfidf_weight=tfidf_weight,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                experiment_name=f"hybrid_weight_comparison_{embedding_weight}_{tfidf_weight}",
                evaluate_metrics=not args.no_metrics
            )
            
            # Store results
            key = f"emb{embedding_weight}_tfidf{tfidf_weight}"
            results[key] = exp_result
        
        print("\nWeight comparison experiments completed!")
    else:
        # Run a single experiment
        run_experiment(
            experiment_type=args.type,
            model_name=args.model,
            embedding_weight=args.embedding_weight,
            tfidf_weight=args.tfidf_weight,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            experiment_name=args.experiment_name,
            evaluate_metrics=not args.no_metrics
        )

if __name__ == "__main__":
    main() 