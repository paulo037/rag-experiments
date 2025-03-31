"""
Data utilities for experiments.
"""

from typing import List, Dict
from rag_testing.core.base import Document

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
        "How do transformers work?": ["doc2", "doc9", "doc8"],  # Transformer-related documents
        "What is RAG?": ["doc3"],  # RAG-related document
        "Tell me about embeddings": ["doc5", "doc9", "doc7", "doc8"],  # Embedding-related documents
        "Explain neural networks": ["doc1"],  # Neural network document
        "Information retrieval with TF-IDF": ["doc4", "doc10"]  # TF-IDF-related documents
    }

def get_standard_test_queries() -> List[str]:
    """Get a standard set of test queries for experiments."""
    return [
        "How do transformers work?",
        "What is RAG?",
        "Tell me about embeddings",
        "Explain neural networks",
        "Information retrieval with TF-IDF"
    ] 