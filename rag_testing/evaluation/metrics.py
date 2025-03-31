from typing import Dict, List, Any, Optional, Set, Callable
import numpy as np
from collections import Counter

from rag_testing.core.base import Document, EvaluationMetric


class PrecisionMetric(EvaluationMetric):
    """Precision evaluation metric for retrieval."""
    
    def evaluate(self, retrieved_docs: List[Document], relevant_docs: List[Document]) -> float:
        """
        Calculate precision of retrieved documents.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Relevant documents (ground truth)
            
        Returns:
            Precision score
        """
        if not retrieved_docs:
            return 0.0
        
        # Get IDs of relevant documents
        relevant_ids = {doc.id for doc in relevant_docs}
        
        # Count how many retrieved documents are relevant
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc.id in relevant_ids)
        
        # Calculate precision
        precision = relevant_retrieved / len(retrieved_docs)
        
        return precision


class RecallMetric(EvaluationMetric):
    """Recall evaluation metric for retrieval."""
    
    def evaluate(self, retrieved_docs: List[Document], relevant_docs: List[Document]) -> float:
        """
        Calculate recall of retrieved documents.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Relevant documents (ground truth)
            
        Returns:
            Recall score
        """
        if not relevant_docs:
            return 1.0  # Perfect recall if there are no relevant documents
        
        if not retrieved_docs:
            return 0.0
        
        # Get IDs of relevant documents
        relevant_ids = {doc.id for doc in relevant_docs}
        
        # Count how many relevant documents were retrieved
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc.id in relevant_ids)
        
        # Calculate recall
        recall = relevant_retrieved / len(relevant_docs)
        
        return recall


class F1Metric(EvaluationMetric):
    """F1 evaluation metric for retrieval."""
    
    def evaluate(self, retrieved_docs: List[Document], relevant_docs: List[Document]) -> float:
        """
        Calculate F1 score for retrieved documents.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Relevant documents (ground truth)
            
        Returns:
            F1 score
        """
        precision_metric = PrecisionMetric()
        recall_metric = RecallMetric()
        
        precision = precision_metric.evaluate(retrieved_docs, relevant_docs)
        recall = recall_metric.evaluate(retrieved_docs, relevant_docs)
        
        if precision + recall == 0:
            return 0.0
        
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1


class NDCGMetric(EvaluationMetric):
    """Normalized Discounted Cumulative Gain metric for retrieval."""
    
    def __init__(self, k: Optional[int] = None):
        """
        Initialize the NDCG metric.
        
        Args:
            k: Cutoff for calculating NDCG@k (None means use all documents)
        """
        self.k = k
    
    def _dcg(self, relevance_scores: List[float]) -> float:
        """Calculate Discounted Cumulative Gain."""
        if not relevance_scores:
            return 0.0
        
        # Apply cutoff if specified
        if self.k is not None:
            relevance_scores = relevance_scores[:self.k]
        
        # Calculate DCG
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
    
    def evaluate(self, retrieved_docs: List[Document], relevant_docs: List[Document]) -> float:
        """
        Calculate NDCG for retrieved documents.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Relevant documents (ground truth)
            
        Returns:
            NDCG score
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        # Create relevance mapping
        relevance_mapping = {}
        for i, doc in enumerate(relevant_docs):
            # Use reverse index as relevance score (higher index = less relevant)
            # Add 1 so all relevant docs have score > 0
            relevance_mapping[doc.id] = len(relevant_docs) - i
        
        # Apply cutoff if specified
        if self.k is not None:
            retrieved_docs = retrieved_docs[:self.k]
        
        # Get relevance scores for retrieved documents
        retrieved_relevance = [relevance_mapping.get(doc.id, 0.0) for doc in retrieved_docs]
        
        # Calculate DCG
        dcg = self._dcg(retrieved_relevance)
        
        # Calculate ideal DCG (best possible ordering)
        ideal_relevance = sorted(relevance_mapping.values(), reverse=True)
        if self.k is not None:
            ideal_relevance = ideal_relevance[:self.k]
        
        idcg = self._dcg(ideal_relevance)
        
        if idcg == 0:
            return 0.0
        
        # Calculate NDCG
        ndcg = dcg / idcg
        
        return ndcg


class MRRMetric(EvaluationMetric):
    """Mean Reciprocal Rank metric for retrieval."""
    
    def evaluate(self, retrieved_docs: List[Document], relevant_docs: List[Document]) -> float:
        """
        Calculate Mean Reciprocal Rank for retrieved documents.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Relevant documents (ground truth)
            
        Returns:
            MRR score
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        # Get IDs of relevant documents
        relevant_ids = {doc.id for doc in relevant_docs}
        
        # Find the rank of the first relevant document
        for i, doc in enumerate(retrieved_docs):
            if doc.id in relevant_ids:
                return 1.0 / (i + 1)
        
        # No relevant documents found
        return 0.0


def get_evaluation_metric(metric_name: str) -> EvaluationMetric:
    """
    Factory function to get an evaluation metric by name.
    
    Args:
        metric_name: Name of the evaluation metric
        
    Returns:
        Evaluation metric instance
    """
    metrics = {
        "precision": PrecisionMetric(),
        "recall": RecallMetric(),
        "f1": F1Metric(),
        "ndcg": NDCGMetric(),
        "mrr": MRRMetric()
    }
    
    # Check for parametrized metrics like ndcg@5
    if "@" in metric_name:
        base_name, k_str = metric_name.split("@")
        try:
            k = int(k_str)
            if base_name == "ndcg":
                return NDCGMetric(k=k)
        except ValueError:
            pass
    
    if metric_name in metrics:
        return metrics[metric_name]
    else:
        raise ValueError(f"Unsupported evaluation metric: {metric_name}") 