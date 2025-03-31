#!/usr/bin/env python
"""
RAG Testing Framework Demo

This script demonstrates the RAG testing framework by running
the embedding + TF-IDF hybrid retrieval example.
"""

import os
import sys
import argparse
from typing import List, Dict, Any

from rag_testing.core.base import Document
from rag_testing.config.models import (
    RAGConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    EvaluationConfig,
    EmbeddingModelType,
    VectorStoreType,
    RetrievalStrategyType
)
from rag_testing.core.pipeline import create_pipeline_from_config
from rag_testing.examples.embedding_tfidf import (
    run_embedding_tfidf_example,
    create_sample_documents,
    create_config
)


def main():
    """Run the RAG testing framework demo."""
    
    parser = argparse.ArgumentParser(description="RAG Testing Framework Demo")
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="hybrid",
        choices=["embedding", "tfidf", "hybrid"],
        help="Retrieval strategy to use"
    )
    parser.add_argument(
        "--embedding-weight", 
        type=float, 
        default=0.7,
        help="Weight for embedding scores in hybrid retrieval"
    )
    parser.add_argument(
        "--tfidf-weight", 
        type=float, 
        default=0.3,
        help="Weight for TF-IDF scores in hybrid retrieval"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model to use"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=3,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--query", 
        type=str,
        help="Custom query to test (optional)"
    )
    
    args = parser.parse_args()
    
    # Create configuration with custom parameters
    config = create_config()
    
    # Update configuration based on arguments
    config.retrieval.strategy_type = args.strategy
    config.retrieval.embedding_weight = args.embedding_weight
    config.retrieval.tfidf_weight = args.tfidf_weight
    config.retrieval.top_k = args.top_k
    config.embedding.model_name = args.model
    
    # Create and set up pipeline
    print(f"Creating RAG pipeline with {args.strategy} retrieval strategy...")
    pipeline = create_pipeline_from_config(config)
    
    # Create and index documents
    print("Creating and indexing sample documents...")
    documents = create_sample_documents()
    pipeline.index(documents)
    
    # Test queries
    default_queries = [
        "What are neural networks?",
        "How do transformers work?",
        "Tell me about embedding models",
        "What is RAG?",
        "Vector search methods"
    ]
    
    # If custom query is provided, use it instead
    if args.query:
        test_queries = [args.query]
        print(f"\nTesting custom query: {args.query}")
    else:
        test_queries = default_queries
        print("\nTesting default queries:")
    
    for query in test_queries:
        if not args.query:  # Only print query for default queries
            print(f"\nQuery: {query}")
        
        results = pipeline.retrieve(query, k=args.top_k)
        
        for i, doc in enumerate(results):
            # Extract scores from metadata
            if args.strategy == "hybrid":
                embedding_score = doc.metadata.get("embedding_score", 0)
                tfidf_score = doc.metadata.get("tfidf_score", 0)
                combined_score = doc.metadata.get("combined_score", 0)
                
                print(f"Result {i+1}: [ID: {doc.id}] (Emb: {embedding_score:.4f}, TFIDF: {tfidf_score:.4f}, Combined: {combined_score:.4f})")
            elif args.strategy == "embedding":
                distance = doc.metadata.get("distance", 0)
                similarity = 1.0 - distance
                print(f"Result {i+1}: [ID: {doc.id}] (Similarity: {similarity:.4f})")
            else:  # tfidf
                similarity = doc.metadata.get("similarity", 0)
                print(f"Result {i+1}: [ID: {doc.id}] (Similarity: {similarity:.4f})")
                
            print(f"  {doc.content}")
    
    # Benchmark performance
    if not args.query:  # Only run benchmark for default queries
        print("\nBenchmarking retrieval performance...")
        benchmark_results = pipeline.benchmark(test_queries)
        
        print(f"\nBenchmark Results:")
        print(f"  Average retrieval time: {benchmark_results['avg_retrieval_time']*1000:.2f} ms")
        print(f"  Number of documents: {benchmark_results['num_documents']}")
        print(f"  Index time: {benchmark_results['index_time']:.2f} s")
    
    print("\nDone!")


if __name__ == "__main__":
    main() 