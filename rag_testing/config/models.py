from pydantic import BaseModel, Field, field_validator, validator
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class EmbeddingModelType(str, Enum):
    """Enumeration of supported embedding model types."""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"
    CUSTOM = "custom"
    MOCK = "mock"


class VectorStoreType(str, Enum):
    """Enumeration of supported vector store types."""
    CHROMA = "chroma"
    CUSTOM = "custom"
    FAISS = "faiss"


class RetrievalStrategyType(str, Enum):
    """Enumeration of supported retrieval strategy types."""
    EMBEDDING = "embedding"
    TFIDF = "tfidf"
    HYBRID = "hybrid"
    CUSTOM = "custom"
    RERANK = "rerank"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    model_type: EmbeddingModelType = Field(...,
                                           description="Type of embedding model")
    model_name: str = Field(..., description="Name of the embedding model")
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model arguments")
    models: Optional[List["EmbeddingConfig"]] = None  # para modelo combinado
    weights: Optional[List[float]] = None


class VectorStoreConfig(BaseModel):
    """Configuration for vector stores."""
    store_type: VectorStoreType
    persist_directory: Optional[str] = None
    collection_name: str = "documents"
    store_kwargs: Optional[Dict[str, Any]] = dict()


class RetrievalConfig(BaseModel):
    """Configuration for retrieval strategies."""
    strategy_type: RetrievalStrategyType
    top_k: int = 5
    embedding_weight: Optional[float] = 0.5
    tfidf_weight: Optional[float] = 0.5
    fusion_method: Optional[str] = "mean"
    strategy_kwargs: Optional[Dict[str, Any]] = dict()

    @field_validator("fusion_method", mode="after")
    def validate_fusion_method(cls, v, values):
        if values.get("strategy_type") == RetrievalStrategyType.HYBRID:
            if v not in {"mean", "max", "weighted"}:
                raise ValueError(
                    "fusion_method must be 'mean', 'max' or 'weighted' for hybrid strategy")
        return v


class DocumentProcessorConfig(BaseModel):
    """Configuration for document processors."""
    chunk_size: int = Field(1000, description="Size of document chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    processors: List[str] = Field(
        default_factory=list, description="List of processors to apply")
    processor_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional processor arguments")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics."""
    metrics: List[str] = Field(default_factory=lambda: ["precision", "recall", "f1"],
                               description="List of evaluation metrics")
    metric_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metric arguments")


class CombinedEmbeddingConfig(BaseModel):
    models: List[EmbeddingConfig]
    weights: Optional[List[float]] = None


class RAGConfig(BaseModel):
    """Configuration for the entire RAG pipeline."""
    type: str
    retrieval: RetrievalConfig
    name: Optional[str] = "default"
    embedding: Optional[Union[EmbeddingConfig, CombinedEmbeddingConfig]] = None
    vector_store: Optional[VectorStoreConfig] = None
    document_processor: Optional[DocumentProcessorConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    description: Optional[str]  = None
