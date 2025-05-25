#!/usr/bin/env python
"""
RAG Testing Framework Demo

This script demonstrates the RAG testing framework by running
the embedding + TF-IDF hybrid retrieval example.
"""

import argparse

from rag_testing.config.models import (
    DocumentProcessorConfig
)
from rag_testing.pipelines.pipeline import create_pipeline_from_config
from rag_testing.data.processors import TextSplitter, TextCleaner, CompositeProcessor
from rag_testing.examples.embedding_tfidf import (
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
    
    # Add document processing options
    parser.add_argument(
        "--process",
        action="store_true",
        help="Enable document processing"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of document chunks"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--clean-text",
        action="store_true",
        help="Enable text cleaning"
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert text to lowercase"
    )
    parser.add_argument(
        "--remove-urls",
        action="store_true",
        help="Remove URLs from text"
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
    
    # Add document processor configuration if enabled
    if args.process:
        config.document_processor = DocumentProcessorConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            processors=["text_cleaner", "text_splitter"] if args.clean_text else ["text_splitter"],
            processor_kwargs={
                "lowercase": args.lowercase,
                "remove_urls": args.remove_urls
            }
        )
    
    # Create and set up pipeline
    print(f"Creating RAG pipeline with {args.strategy} retrieval strategy...")
    pipeline = create_pipeline_from_config(config)
    
    # If processing is enabled, create document processors manually
    # This demonstrates how to create processors outside of the config
    if args.process:
        processors = []
        
        if args.clean_text:
            processors.append(TextCleaner(
                lowercase=args.lowercase,
                remove_urls=args.remove_urls
            ))
        
        processors.append(TextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        ))
        
        document_processor = CompositeProcessor(processors)
        pipeline.document_processor = document_processor
        
        print("Document processing enabled:")
        print(f"  - Chunk size: {args.chunk_size}")
        print(f"  - Chunk overlap: {args.chunk_overlap}")
        if args.clean_text:
            print("  - Text cleaning enabled")
            if args.lowercase:
                print("  - Lowercase conversion enabled")
            if args.remove_urls:
                print("  - URL removal enabled")
    
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
        
        print("\nBenchmark Results:")
        print(f"  Average retrieval time: {benchmark_results['avg_retrieval_time']*1000:.2f} ms")
        print(f"  Number of documents: {benchmark_results['num_documents']}")
        print(f"  Index time: {benchmark_results['index_time']:.2f} s")
    
    print("\nDone!")


if __name__ == "__main__":
    main() 