from typing import Dict, List, Any, Optional
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
    type: str = "simple"
    
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
        if self.document_processor:
            return self.document_processor.process(documents)
        return documents
    
    def index(self, documents: List[Document]) -> None:
        start_time = time.time()
        
        processed_docs = self.process_documents(documents)
        
        self.retrieval_strategy.setup(processed_docs)
        
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
        if not self.indexed:
            raise ValueError("Documents must be indexed before retrieval")
        
        start_time = time.time()
        
        
        doc_results = self.retrieval_strategy.retrieve(query, k=k)
        
        results = []
        for doc in doc_results:
            if "combined_score" in doc.metadata:
                score_field = "combined_score"
            elif "similarity" in doc.metadata:
                score_field = "similarity"
            else:
                score_field = "distance"
        
            result = SearchResult.from_document(doc, score_field=score_field)
            results.append(result)
        
        self.statistics["retrieval_times"].append(time.time() - start_time)
        
        return results
    
    def evaluate(self, queries: List[str], relevant_docs: Dict[str, List[Document]]) -> Dict[str, float]:
        if not self.evaluation_metrics:
            return {}
        
        results = {metric.__class__.__name__: 0.0 for metric in self.evaluation_metrics}
        
        for query in tqdm(queries, desc="Evaluating"):
            if query not in relevant_docs:
                continue
                       
            search_results = self.retrieve(query)
            retrieved_docs = [result.document for result in search_results]
                      
            for metric in self.evaluation_metrics:
                metric_name = metric.__class__.__name__
                score = metric.evaluate(retrieved_docs, relevant_docs[query])
                             
                results[metric_name] += score
             
        for metric_name in results:
            results[metric_name] /= len(queries)
              
        self.statistics["evaluation_results"] = results
        
        return results
    
    def benchmark(self, queries: List[str], k: int = 5) -> Dict[str, Any]:
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
        stats = dict(self.statistics)
        
        retrieval_times = stats.get("retrieval_times", [])
        if retrieval_times:
            stats["avg_retrieval_time"] = sum(retrieval_times) / len(retrieval_times)
        
        return stats

    @classmethod
    def build(cls, config: RAGConfig) -> "RAGPipeline":

        embedding_model = get_embedding_model(config.embedding) if config.embedding else None
        vector_store = get_vector_store(config.vector_store) if config.vector_store else None
        retrieval_strategy = get_retrieval_strategy(
            config.retrieval, 
            embedding_model, 
            vector_store
        )
        
        
        evaluation_metrics = []
        if config.evaluation:
            for metric_name in config.evaluation.metrics:
                evaluation_metrics.append(get_evaluation_metric(metric_name))
        
        
        pipeline = cls(
            embedding_model=embedding_model,
            vector_store=vector_store,
            retrieval_strategy=retrieval_strategy,
            evaluation_metrics=evaluation_metrics
        )
        
        return pipeline 


