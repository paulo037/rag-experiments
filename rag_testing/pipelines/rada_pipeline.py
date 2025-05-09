from rag_testing.core.base import RAGPipeline
from rag_testing.core.pipeline import SimpleRAGPipeline
from rag_testing.embeddings.models import get_embedding_model, CombinedEmbeddingModel
from rag_testing.retrieval.vector_stores import get_vector_store
from rag_testing.retrieval.strategies import get_retrieval_strategy
from rag_testing.evaluation.metrics import get_evaluation_metric
from rag_testing.config.models import RAGConfig, EmbeddingConfig, CombinedEmbeddingConfig


def create_rada_pipeline(config: RAGConfig) -> RAGPipeline:
    """
    Creates a RADA pipeline with specific configuration.
    
    Args:
        config: RAGConfig object
    
    Returns:
        A RAGPipeline (customized for RADA)
    """
    # Detecta se estamos lidando com múltiplos modelos
    if isinstance(config.embedding, CombinedEmbeddingConfig):
        embedding_models = [
            get_embedding_model(model_cfg) for model_cfg in config.embedding.models
        ]
        embedding_model = CombinedEmbeddingModel(
            models=embedding_models,
            weights=config.embedding.weights
        )
    else:
        embedding_model = get_embedding_model(config.embedding)

    vector_store = get_vector_store(config.vector_store)
    retrieval_strategy = get_retrieval_strategy(config.retrieval, embedding_model, vector_store)

    evaluation_metrics = []
    if config.evaluation:
        for metric_name in config.evaluation.metrics:
            evaluation_metrics.append(get_evaluation_metric(metric_name))

    pipeline = SimpleRAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        retrieval_strategy=retrieval_strategy,
        evaluation_metrics=evaluation_metrics
    )
    
    return pipeline
