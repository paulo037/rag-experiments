"""
Pipeline configuration factories for different experiment types.
"""

from typing import Dict, Any
from rag_testing.config.models import (
    RAGConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    EmbeddingModelType,
    VectorStoreType,
    RetrievalStrategyType
)

def create_embedding_config(
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5
) -> RAGConfig:
    """Create a configuration for pure embedding-based retrieval."""
    return RAGConfig(
        type="embedding",
        embedding=EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name=model_name
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStoreType.CHROMA,
            collection_name="embedding_experiment"
        ),
        retrieval=RetrievalConfig(
            strategy_type=RetrievalStrategyType.EMBEDDING,
            top_k=top_k
        )
    )

def create_hybrid_config(
    model_name: str = "all-MiniLM-L6-v2",
    embedding_weight: float = 0.7,
    tfidf_weight: float = 0.3,
    top_k: int = 5
) -> RAGConfig:
    """Create a configuration for hybrid embedding + TF-IDF retrieval."""
    return RAGConfig(
        type="hybrid",
        embedding=EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name=model_name
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStoreType.CHROMA,
            collection_name="hybrid_experiment"
        ),
        retrieval=RetrievalConfig(
            strategy_type=RetrievalStrategyType.HYBRID,
            top_k=top_k,
            embedding_weight=embedding_weight,
            tfidf_weight=tfidf_weight
        )
    )

def create_tfidf_config(top_k: int = 5) -> RAGConfig:
    """Create a configuration for pure TF-IDF retrieval."""
    return RAGConfig(
        type="tfidf",
        retrieval=RetrievalConfig(
            strategy_type=RetrievalStrategyType.TFIDF,
            top_k=top_k
        )
    )

def get_experiment_config(
    experiment_type: str,
    **kwargs
) -> RAGConfig:
    """
    Factory function to create a configuration for a specific experiment type.
    
    Args:
        experiment_type: Type of experiment ('embedding', 'hybrid', or 'tfidf')
        **kwargs: Additional configuration parameters
        
    Returns:
        RAG configuration for the specified experiment type
    """
    config_factories = {
        "embedding": create_embedding_config,
        "hybrid": create_hybrid_config,
        "tfidf": create_tfidf_config
    }
    
    if experiment_type not in config_factories and experiment_type != "rada":
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    if experiment_type == "rada":
        return RAGConfig(
            type="rada",
            embedding={"model_name": kwargs.get("model_name", "all-MiniLM-L6-v2")},
            vector_store={"type": "faiss"},
            retrieval={"strategy": "rerank", "top_k": kwargs.get("top_k", 5)},
            evaluation={"metrics": ["Recall", "Precision"]},
            chunking={"chunk_size": kwargs.get("chunk_size", 200), "chunk_overlap": kwargs.get("chunk_overlap", 50)}
        )

    return config_factories[experiment_type](**kwargs)
