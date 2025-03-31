"""
Basic tests for the RAG pipeline.
"""

import os
import sys
import unittest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from rag_testing.embeddings.models import MockEmbedding
from rag_testing.retrieval.vector_stores import ChromaVectorStore
from rag_testing.retrieval.strategies import EmbeddingRetrieval, TfidfRetrieval, HybridRetrieval


class TestBasicPipeline(unittest.TestCase):
    """Test basic functionality of the RAG pipeline."""
    
    def setUp(self):
        """Set up test documents."""
        self.documents = [
            Document(
                content="Neural networks are a class of machine learning algorithms inspired by the human brain.",
                metadata={"category": "machine_learning", "source": "sample"},
                id="doc1"
            ),
            Document(
                content="Transformers have revolutionized natural language processing with attention mechanisms.",
                metadata={"category": "nlp", "source": "sample"},
                id="doc2"
            ),
            Document(
                content="Retrieval-Augmented Generation (RAG) combines retrieval systems with text generation.",
                metadata={"category": "rag", "source": "sample"},
                id="doc3"
            )
        ]
    
    def test_embedding_retrieval(self):
        """Test embedding-based retrieval."""
        # Create a configuration with mock embedding model
        config = RAGConfig(
            name="test_embedding",
            embedding=EmbeddingConfig(
                model_type=EmbeddingModelType.MOCK,
                model_name="mock",
                model_kwargs={"embedding_dim": 384}
            ),
            vector_store=VectorStoreConfig(
                store_type=VectorStoreType.CHROMA,
                collection_name="test_embedding"
            ),
            retrieval=RetrievalConfig(
                strategy_type=RetrievalStrategyType.EMBEDDING,
                top_k=3
            )
        )
        
        # Create pipeline
        pipeline = create_pipeline_from_config(config)
        
        # Index documents
        pipeline.index(self.documents)
        
        # Test retrieval
        results = pipeline.retrieve("neural networks", k=2)
        
        # Check results
        self.assertEqual(len(results), 2, "Should retrieve 2 documents")
        
        # Test that retrieval time is recorded
        self.assertGreater(len(pipeline.statistics["retrieval_times"]), 0)
    
    def test_tfidf_retrieval(self):
        """Test TF-IDF retrieval."""
        # Create a configuration with TF-IDF retrieval
        config = RAGConfig(
            name="test_tfidf",
            embedding=EmbeddingConfig(
                model_type=EmbeddingModelType.MOCK,
                model_name="mock",
                model_kwargs={"embedding_dim": 384}
            ),
            vector_store=VectorStoreConfig(
                store_type=VectorStoreType.CHROMA,
                collection_name="test_tfidf"
            ),
            retrieval=RetrievalConfig(
                strategy_type=RetrievalStrategyType.TFIDF,
                top_k=3
            )
        )
        
        # Create pipeline
        pipeline = create_pipeline_from_config(config)
        
        # Index documents
        pipeline.index(self.documents)
        
        # Test retrieval
        results = pipeline.retrieve("neural networks", k=2)
        
        # Check results
        self.assertEqual(len(results), 2, "Should retrieve 2 documents")
        
        # Check that at least one result has "neural" in it
        has_neural = any("neural" in doc.content.lower() for doc in results)
        self.assertTrue(has_neural, "Should retrieve documents about neural networks")
    
    def test_hybrid_retrieval(self):
        """Test hybrid retrieval."""
        # Create a configuration with hybrid retrieval
        config = RAGConfig(
            name="test_hybrid",
            embedding=EmbeddingConfig(
                model_type=EmbeddingModelType.MOCK,
                model_name="mock",
                model_kwargs={"embedding_dim": 384}
            ),
            vector_store=VectorStoreConfig(
                store_type=VectorStoreType.CHROMA,
                collection_name="test_hybrid"
            ),
            retrieval=RetrievalConfig(
                strategy_type=RetrievalStrategyType.HYBRID,
                top_k=3,
                embedding_weight=0.5,
                tfidf_weight=0.5
            )
        )
        
        # Create pipeline
        pipeline = create_pipeline_from_config(config)
        
        # Index documents
        pipeline.index(self.documents)
        
        # Test retrieval
        results = pipeline.retrieve("neural networks", k=2)
        
        # Check results
        self.assertEqual(len(results), 2, "Should retrieve 2 documents")
        
        # Check for scores in metadata
        first_result = results[0]
        self.assertIn("embedding_score", first_result.metadata)
        self.assertIn("tfidf_score", first_result.metadata)
        self.assertIn("combined_score", first_result.metadata)


if __name__ == "__main__":
    unittest.main() 