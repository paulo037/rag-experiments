#!/usr/bin/env python
"""
Experiment Runner Module

This module provides a generic experiment runner for evaluating different RAG pipelines.
It handles experiment setup, running, metrics collection, and result saving.
"""

import os
import json
import csv
import platform
import time
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

import mlflow
from sentence_transformers import SentenceTransformer
import tiktoken
import sklearn

from rag_testing.core.base import Document, SearchResult
from rag_testing.pipelines.pipeline import RAGPipeline
from rag_testing.data.processors import TextSplitter, CompositeProcessor
from rag_testing.evaluation.metrics import (
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    NDCGMetric,
    MRRMetric
)


def create_test_documents() -> List[Document]:
    """Create a set of test documents."""
    return [
        Document(
            content="Neural networks are a class of machine learning algorithms inspired by the human brain.",
            id="doc1",
            metadata={"source": "test_data", "topic": "neural_networks"}
        ),
        Document(
            content="Transformers have revolutionized natural language processing with attention mechanisms.",
            id="doc2",
            metadata={"source": "test_data", "topic": "transformers"}
        ),
        Document(
            content="Retrieval-Augmented Generation (RAG) combines retrieval systems with text generation.",
            id="doc3",
            metadata={"source": "test_data", "topic": "rag"}
        ),
        Document(
            content="TF-IDF stands for Term Frequency-Inverse Document Frequency, a numerical statistic used in information retrieval.",
            id="doc4",
            metadata={"source": "test_data", "topic": "tfidf"}
        ),
        Document(
            content="Embedding models map text to high-dimensional vector spaces to capture semantic meaning.",
            id="doc5",
            metadata={"source": "test_data", "topic": "embeddings"}
        ),
        Document(
            content="Cosine similarity measures the cosine of the angle between two vectors in a vector space.",
            id="doc6",
            metadata={"source": "test_data", "topic": "vector_similarity"}
        ),
        Document(
            content="Vector databases store and index vector embeddings for efficient similarity search.",
            id="doc7",
            metadata={"source": "test_data", "topic": "vector_databases"}
        ),
        Document(
            content="The all-MiniLM-L6-v2 model is a small, efficient transformer model for generating sentence embeddings.",
            id="doc8",
            metadata={"source": "test_data", "topic": "sentence_transformers"}
        ),
        Document(
            content="Sentence transformers are models specifically designed to create embeddings for sentences and paragraphs.",
            id="doc9",
            metadata={"source": "test_data", "topic": "sentence_transformers"}
        ),
        Document(
            content="TF-IDF helps find relevant documents by weighting terms based on their frequency in documents.",
            id="doc10",
            metadata={"source": "test_data", "topic": "tfidf"}
        )
    ]


def create_test_query_relevance() -> Dict[str, List[str]]:
    """
    Create test queries with relevant document IDs.
    
    Returns:
        Dictionary mapping queries to lists of relevant document IDs
    """
    return {
        "How do transformers work?": ["doc2", "doc9", "doc8"],  
        "What is RAG?": ["doc3"],  
        "Tell me about embeddings": ["doc5", "doc9", "doc7", "doc8"],  
        "Explain neural networks": ["doc1"],  
        "Information retrieval with TF-IDF": ["doc4", "doc10"]  
    }


def evaluate_search_results(
    query: str,
    search_results: List[SearchResult],
    relevant_doc_ids: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Evaluate search results using standard IR metrics.
    
    Args:
        query: The search query
        search_results: List of search results
        relevant_doc_ids: Dictionary mapping queries to relevant document IDs
        
    Returns:
        Dictionary of evaluation metrics
    """
    if query not in relevant_doc_ids:
        return {}
    
    
    expected_ids = relevant_doc_ids[query]
    
    
    retrieved_docs = [result.document for result in search_results]
    
    
    
    relevant_docs = [
        Document(content="", id=doc_id, metadata={})
        for doc_id in expected_ids
    ]
    
    
    precision = PrecisionMetric().evaluate(retrieved_docs, relevant_docs)
    recall = RecallMetric().evaluate(retrieved_docs, relevant_docs)
    f1 = F1Metric().evaluate(retrieved_docs, relevant_docs)
    ndcg = NDCGMetric().evaluate(retrieved_docs, relevant_docs)
    mrr = MRRMetric().evaluate(retrieved_docs, relevant_docs)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ndcg": ndcg,
        "mrr": mrr
    }


def get_model_details(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about the embedding model.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        Dictionary containing model details
    """
    try:
        
        model = SentenceTransformer(model_name)
        
        
        modules = []
        for i, module in enumerate(model.modules()):
            if i > 1:  
                module_name = module.__class__.__name__
                modules.append(module_name)
        
        
        params = sum(p.numel() for p in model.parameters())
        
        return {
            "model_name": model_name,
            "model_type": "SentenceTransformer",
            "vector_dim": model.get_sentence_embedding_dimension(),
            "parameters": params,
            "modules": modules,
            "is_multilingual": "multilingual" in model_name.lower(),
            "tokenizer": str(model.tokenizer),
            "max_seq_length": model.max_seq_length,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sentence_transformers_version": SentenceTransformer.__version__
        }
    except Exception as e:
        print(f"Error getting model details: {e}")
        return {"model_name": model_name, "error": str(e)}


def get_tokenizer_details(encoding_name: str) -> Dict[str, Any]:
    """
    Get detailed information about the tokenizer used for chunking.
    
    Args:
        encoding_name: Name of the tiktoken encoding
        
    Returns:
        Dictionary containing tokenizer details
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return {
            "tokenizer_name": encoding_name,
            "tokenizer_type": "tiktoken",
            "tiktoken_version": tiktoken.__version__,
            "vocabulary_size": encoding.n_vocab
        }
    except Exception as e:
        print(f"Error getting tokenizer details: {e}")
        return {"tokenizer_name": encoding_name, "error": str(e)}


def get_tfidf_details() -> Dict[str, Any]:
    """
    Get detailed information about the TF-IDF configuration.
    
    Returns:
        Dictionary containing TF-IDF details
    """
    return {
        "tfidf_implementation": "scikit-learn",
        "scikit_learn_version": sklearn.__version__,
        "analyzer": "word",
        "stop_words": "english",
        "ngram_range": "1-1",
        "max_df": 0.95,
        "min_df": 2
    }


def run_retrieval_experiment(
    pipeline: RAGPipeline,
    test_queries: List[str],
    chunk_size: int,
    chunk_overlap: int,
    run_name: Optional[str] = None,
    experiment_name: str = "retrieval_experiments",
    display_results: bool = True,
    result_formatter: Optional[Callable[[SearchResult], str]] = None,
    evaluate_metrics: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a generic retrieval experiment using the provided pipeline.
    
    Args:
        pipeline: The RAG pipeline to test
        config_params: Dictionary of configuration parameters to log
        test_queries: List of queries to test
        chunk_size: Size of document chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        output_dir: Directory to save experiment results
        run_name: Custom name for the MLflow run
        experiment_name: Name of the MLflow experiment
        display_results: Whether to print results to console
        result_formatter: Optional function to format search results for display
        evaluate_metrics: Whether to calculate evaluation metrics
        
    Returns:
        Dictionary of experiment results
    """
    
    mlflow.set_experiment(experiment_name)
    
    
    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"retrieval_experiment_{timestamp}"
    
    
    relevant_doc_ids = create_test_query_relevance() if evaluate_metrics else {}
    
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        
        
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
        
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("platform", platform.platform())
        
        
        tokenizer_details = get_tokenizer_details("cl100k_base")
        for key, value in tokenizer_details.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"tokenizer_{key}", value)
        
        
        document_processor = CompositeProcessor([
            TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name="cl100k_base")
        ])
        pipeline.document_processor = document_processor
        
        
        documents = create_test_documents()
        
        
        processed_docs = pipeline.process_documents(documents)
        start_time = time.time()
        pipeline.ingest_documents(processed_docs)
        index_time = time.time() - start_time
        
        
        mlflow.log_metric("index_time_seconds", index_time)
        mlflow.log_metric("num_source_documents", len(documents))
        mlflow.log_metric("num_chunks", len(processed_docs))
        
        
        results = []
        retrieval_times = []
        metrics_by_query = {}
        overall_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0
        }
        
        for query in test_queries:
            start_time = time.time()
            search_results = pipeline.retrieve(query)
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time * 1000)  
            
            
            query_metrics = {}
            if evaluate_metrics and query in relevant_doc_ids:
                query_metrics = evaluate_search_results(query, search_results, relevant_doc_ids)
                metrics_by_query[query] = query_metrics
                
                
                for metric_name, value in query_metrics.items():
                    mlflow.log_metric(f"query_{test_queries.index(query)+1}_{metric_name}", value)
                    overall_metrics[metric_name] += value
            
            
            formatted_results = []
            for result in search_results:
                if result_formatter:
                    formatted_result = result_formatter(result)
                else:
                    
                    formatted_result = f"[ID: {result.document.id}] (Similarity: {result.score:.4f})\n  {result.document.content}"
                formatted_results.append(formatted_result)
            
            results.append({
                "query": query,
                "results": search_results,
                "formatted_results": formatted_results,
                "retrieval_time_ms": retrieval_time * 1000,
                "metrics": query_metrics
            })
            
            if display_results:
                print(f"\nQuery: {query}")
                for i, formatted_result in enumerate(formatted_results, 1):
                    print(f"Result {i}: {formatted_result}")
                
                
                if query_metrics:
                    print(f"Metrics: {', '.join([f'{k}={v:.4f}' for k, v in query_metrics.items()])}")
        
        
        if evaluate_metrics and test_queries:
            num_evaluated_queries = sum(1 for q in test_queries if q in relevant_doc_ids)
            if num_evaluated_queries > 0:
                for metric in overall_metrics:
                    overall_metrics[metric] /= num_evaluated_queries
                    mlflow.log_metric(metric, overall_metrics[metric])
        
        
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        
        
        mlflow.log_metric("avg_retrieval_time_ms", avg_retrieval_time)
        
        
        if display_results:
            print("\nBenchmark Results:")
            print(f"  Average retrieval time: {avg_retrieval_time:.2f} ms")
            print(f"  Number of documents: {len(documents)}")
            print(f"  Index time: {index_time:.2f} s")
            
            
            if evaluate_metrics:
                print("\nOverall Metrics:")
                for metric, value in overall_metrics.items():
                    print(f"  {metric}: {value:.4f}")
        
        
        experiment_results = {
            "experiment_info": {
                "run_id": run_id,
                "experiment_name": experiment_name,
                "run_name": run_name,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "metrics": {
                "index_time_seconds": index_time,
                "avg_retrieval_time_ms": avg_retrieval_time,
                "num_source_documents": len(documents),
                "num_chunks": len(processed_docs),
                **overall_metrics
            },
            "queries": []
        }
        
        
        for query_result in results:
            serializable_results = []
            for result in query_result["results"]:
                serializable_result = {
                    "document_id": result.document.id,
                    "content": result.document.content,
                    "metadata": result.document.metadata,
                    "score": result.score
                }
                
                if hasattr(result, "embedding_score") and result.embedding_score is not None:
                    serializable_result["embedding_score"] = result.embedding_score
                if hasattr(result, "tfidf_score") and result.tfidf_score is not None:
                    serializable_result["tfidf_score"] = result.tfidf_score
                
                serializable_results.append(serializable_result)
            
            experiment_results["queries"].append({
                "query": query_result["query"],
                "results": serializable_results,
                "retrieval_time_ms": query_result["retrieval_time_ms"],
                "metrics": query_result.get("metrics", {})
            })
        
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_json:
            json.dump(experiment_results, temp_json, indent=2)
            temp_json_path = temp_json.name
        
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_csv:
            writer = csv.writer(temp_csv)
            
            
            header = ["query", "result_rank", "document_id", "score", "content"]
            
            
            metric_columns = []
            if any(query_result.get("metrics") for query_result in results):
                metric_names = next((query_result["metrics"].keys() for query_result in results if query_result.get("metrics")), [])
                metric_columns = [f"query_{metric}" for metric in metric_names]
                header.extend(metric_columns)
            
            
            if any(hasattr(results[0]["results"][0], attr) and getattr(results[0]["results"][0], attr) is not None 
                   for attr in ["embedding_score", "tfidf_score"]):
                header.insert(4, "tfidf_score")
                header.insert(4, "embedding_score")
            
            writer.writerow(header)
            
            
            for query_result in experiment_results["queries"]:
                query = query_result["query"]
                query_metrics = query_result.get("metrics", {})
                
                for i, result in enumerate(query_result["results"], 1):
                    row = [query, i, result["document_id"], result["score"], result["content"]]
                    
                    
                    if "embedding_score" in result and "tfidf_score" in result:
                        row.insert(4, result["tfidf_score"])
                        row.insert(4, result["embedding_score"])
                    
                    
                    if metric_columns:
                        for metric in metric_names:
                            row.append(query_metrics.get(metric, ""))
                    
                    writer.writerow(row)
            
            temp_csv_path = temp_csv.name
        
        
        try:
            mlflow.log_artifact(temp_json_path)
            mlflow.log_artifact(temp_csv_path)
            
            if display_results:
                print("\nExperiment results logged to MLflow")
                print(f"MLflow run ID: {run_id}")
                print("\nExperiment completed!")
        finally:
            
            if os.path.exists(temp_json_path):
                os.unlink(temp_json_path)
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
        
        return experiment_results 