from typing import Dict, List, Any, Optional, Union
import pandas as pd
import time
from tqdm import tqdm

from rag_testing.core.base import (
    Document, 
    EmbeddingModel, 
    VectorStore, 
    RetrievalStrategy, 
    DocumentProcessor,
    EvaluationMetric,
    RAGPipeline
)
from rag_testing.config.models import RAGConfig
from rag_testing.embeddings.models import get_embedding_model
from rag_testing.retrieval.vector_stores import get_vector_store
from rag_testing.retrieval.strategies import get_retrieval_strategy
from rag_testing.evaluation.metrics import get_evaluation_metric


class SimpleRAGPipeline(RAGPipeline):
    """Simple RAG pipeline implementation."""
    
    def __init__(self, 
                 embedding_model: EmbeddingModel,
                 vector_store: VectorStore,
                 retrieval_strategy: RetrievalStrategy,
                 document_processor: Optional[DocumentProcessor] = None,
                 evaluation_metrics: Optional[List[EvaluationMetric]] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Model for generating embeddings
            vector_store: Vector store for similarity search
            retrieval_strategy: Strategy for retrieving documents
            document_processor: Processor for documents (optional)
            evaluation_metrics: List of evaluation metrics (optional)
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.retrieval_strategy = retrieval_strategy
        self.document_processor = document_processor
        self.evaluation_metrics = evaluation_metrics or []
        self.indexed = False
        self.statistics = {
            "index_time": 0,
            "num_documents": 0,
            "retrieval_times": [],
            "evaluation_results": {}
        }
    
    def index(self, documents: List[Document]) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of documents to index
        """
        start_time = time.time()
        
        # Process documents if processor is provided
        processed_docs = documents
        if self.document_processor:
            processed_docs = self.document_processor.process(documents)
        
        # Set up the retrieval strategy with processed documents
        self.retrieval_strategy.setup(processed_docs)
        
        # Update statistics
        self.statistics["index_time"] = time.time() - start_time
        self.statistics["num_documents"] = len(processed_docs)
        self.indexed = True
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if not self.indexed:
            raise ValueError("Documents must be indexed before retrieval")
        
        start_time = time.time()
        
        # Use the retrieval strategy to get documents
        results = self.retrieval_strategy.retrieve(query, k=k)
        
        # Update statistics
        self.statistics["retrieval_times"].append(time.time() - start_time)
        
        return results
    
    def evaluate(self, queries: List[str], relevant_docs: Dict[str, List[Document]]) -> Dict[str, float]:
        """
        Evaluate the pipeline on a set of queries.
        
        Args:
            queries: List of query strings
            relevant_docs: Dictionary mapping queries to relevant documents
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.evaluation_metrics:
            return {}
        
        results = {metric.__class__.__name__: 0.0 for metric in self.evaluation_metrics}
        
        for query in tqdm(queries, desc="Evaluating"):
            if query not in relevant_docs:
                continue
            
            # Retrieve documents for the query
            retrieved_docs = self.retrieve(query)
            
            # Calculate metrics
            for metric in self.evaluation_metrics:
                metric_name = metric.__class__.__name__
                score = metric.evaluate(retrieved_docs, relevant_docs[query])
                
                # Accumulate scores
                results[metric_name] += score
        
        # Average the scores
        for metric_name in results:
            results[metric_name] /= len(queries)
        
        # Store evaluation results
        self.statistics["evaluation_results"] = results
        
        return results
    
    def benchmark(self, queries: List[str], k: int = 5) -> Dict[str, Any]:
        """
        Benchmark retrieval performance.
        
        Args:
            queries: List of query strings to benchmark
            k: Number of documents to retrieve
            
        Returns:
            Dictionary of benchmark statistics
        """
        if not self.indexed:
            raise ValueError("Documents must be indexed before benchmarking")
        
        retrieval_times = []
        
        for query in tqdm(queries, desc="Benchmarking"):
            start_time = time.time()
            _ = self.retrieve(query, k=k)
            retrieval_times.append(time.time() - start_time)
        
        benchmark_results = {
            "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times),
            "min_retrieval_time": min(retrieval_times),
            "max_retrieval_time": max(retrieval_times),
            "num_queries": len(queries),
            "index_time": self.statistics["index_time"],
            "num_documents": self.statistics["num_documents"]
        }
        
        return benchmark_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = dict(self.statistics)
        
        # Add average retrieval time if available
        retrieval_times = stats.get("retrieval_times", [])
        if retrieval_times:
            stats["avg_retrieval_time"] = sum(retrieval_times) / len(retrieval_times)
        
        return stats


def create_pipeline_from_config(config: RAGConfig) -> RAGPipeline:
    """
    Create a RAG pipeline from configuration.
    
    Args:
        config: RAG configuration
        
    Returns:
        Configured RAG pipeline
    """
    # Create embedding model
    embedding_model = get_embedding_model(config.embedding)
    
    # Create vector store
    vector_store = get_vector_store(config.vector_store)
    
    # Create retrieval strategy
    retrieval_strategy = get_retrieval_strategy(
        config.retrieval, 
        embedding_model, 
        vector_store
    )
    
    # Create evaluation metrics if configured
    evaluation_metrics = []
    if config.evaluation:
        for metric_name in config.evaluation.metrics:
            evaluation_metrics.append(get_evaluation_metric(metric_name))
    
    # Create pipeline
    pipeline = SimpleRAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        retrieval_strategy=retrieval_strategy,
        evaluation_metrics=evaluation_metrics
    )
    
    return pipeline 