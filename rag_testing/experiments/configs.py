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
    top_k: int = 5,
    **kwargs
) -> RAGConfig:
    """Create a configuration for pure embedding-based retrieval."""
    return RAGConfig(
        type="simple",
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
        ),
        document_processor=None,
        evaluation=None,
        name=None,
        description="Embedding experiment"
    )

def create_hybrid_config(
    model_name: str = "all-MiniLM-L6-v2",
    embedding_weight: float = 0.7,
    tfidf_weight: float = 0.3,
    top_k: int = 5,
    **kwargs
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

def create_tfidf_config(
    top_k: int = 5,
    **kwargs
) -> RAGConfig:
    """Create a configuration for pure TF-IDF retrieval."""
    return RAGConfig(
        type="tfidf",
        retrieval=RetrievalConfig(
            strategy_type=RetrievalStrategyType.TFIDF,
            top_k=top_k
        )
    )

def create_rada_config(
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    **kwargs
) -> RAGConfig:
    return RAGConfig(
            type="rada",
            embedding=EmbeddingConfig(
                model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
                model_name=model_name
            ),
            vector_store=VectorStoreConfig(
                store_type=VectorStoreType.CHROMA
            ),
            retrieval=RetrievalConfig(
                strategy_type=RetrievalStrategyType.EMBEDDING,
                top_k=top_k,
            )
        )

def get_experiment_config(
    config_type: str,
    **kwargs
) -> RAGConfig:

    config_factories = {
        "embedding": create_embedding_config,
        "hybrid": create_hybrid_config,
        "tfidf": create_tfidf_config,
        "rada": create_rada_config
    }
    
    assert config_type in config_factories, f"Unknown experiment type: {config_type}"
    
    return config_factories[config_type](**kwargs)
