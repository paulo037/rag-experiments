from typing import Dict, List, Any, Optional, Union
import pandas as pd
import time
from tqdm import tqdm

from rag_testing.core.base import (
    Document, 
    SearchResult,
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
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents if a document processor is available.
        
        Args:
            documents: Documents to process
            
        Returns:
            Processed documents
        """
        if self.document_processor:
            return self.document_processor.process(documents)
        return documents
    
    def index(self, documents: List[Document]) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of documents to index
        """
        start_time = time.time()
        
        # Process documents if processor is provided
        processed_docs = self.process_documents(documents)
        
        # Set up the retrieval strategy with processed documents
        self.retrieval_strategy.setup(processed_docs)
        
        # Update statistics
        self.statistics["index_time"] = time.time() - start_time
        self.statistics["num_documents"] = len(processed_docs)
        self.indexed = True
    
    def ingest_documents(self, documents: List[Document]) -> None:
        """
        Ingest documents into the pipeline.
        Alias for index() method.
        
        Args:
            documents: Documents to ingest
        """
        self.index(documents)
    
    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of search results with documents and scores
        """
        if not self.indexed:
            raise ValueError("Documents must be indexed before retrieval")
        
        start_time = time.time()
        
        # Use the retrieval strategy to get documents
        doc_results = self.retrieval_strategy.retrieve(query, k=k)
        
        # Convert to SearchResult objects
        results = []
        for doc in doc_results:
            # Determine score field based on retrieval strategy type
            if "combined_score" in doc.metadata:
                score_field = "combined_score"
            elif "similarity" in doc.metadata:
                score_field = "similarity"
            else:
                score_field = "distance"
                
            # Create SearchResult from Document
            result = SearchResult.from_document(doc, score_field=score_field)
            results.append(result)
        
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
            search_results = self.retrieve(query)
            retrieved_docs = [result.document for result in search_results]
            
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
    embedding_model = get_embedding_model(config.embedding) if config.embedding else None
    
    # Create vector store
    vector_store = get_vector_store(config.vector_store) if config.vector_store else None
    
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
    
        # pipeline RADA
    if config.type == "rada":
        from rag_testing.pipelines.rada_pipeline import create_rada_pipeline
        return create_rada_pipeline(config)

    
    return pipeline 