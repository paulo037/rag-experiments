from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional, Union, Type
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



class RetrievalStrategyType(str, Enum):
    """Enumeration of supported retrieval strategy types."""
    EMBEDDING = "embedding"
    TFIDF = "tfidf"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    model_type: EmbeddingModelType = Field(..., description="Type of embedding model")
    model_name: str = Field(..., description="Name of the embedding model")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional model arguments")


class VectorStoreConfig(BaseModel):
    """Configuration for vector stores."""
    store_type: VectorStoreType = Field(..., description="Type of vector store")
    persist_directory: Optional[str] = Field(None, description="Directory to persist the vector store")
    collection_name: str = Field("documents", description="Name of the collection")
    store_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional vector store arguments")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval strategies."""
    strategy_type: RetrievalStrategyType = Field(..., description="Type of retrieval strategy")
    top_k: int = Field(5, description="Number of documents to retrieve")
    
    # For hybrid retrieval
    embedding_weight: Optional[float] = Field(0.5, description="Weight for embedding scores in hybrid retrieval")
    tfidf_weight: Optional[float] = Field(0.5, description="Weight for TF-IDF scores in hybrid retrieval")
    
    strategy_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional strategy arguments")


class DocumentProcessorConfig(BaseModel):
    """Configuration for document processors."""
    chunk_size: int = Field(1000, description="Size of document chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    processors: List[str] = Field(default_factory=list, description="List of processors to apply")
    processor_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional processor arguments")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics."""
    metrics: List[str] = Field(default_factory=lambda: ["precision", "recall", "f1"], 
                              description="List of evaluation metrics")
    metric_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional metric arguments")


class RAGConfig(BaseModel):
    """Configuration for the entire RAG pipeline."""
    type: Optional[str] = Field(None, description="Type of RAG pipeline (e.g., 'rada', 'tfidf', 'embedding')")

    
    embedding: Optional[EmbeddingConfig] = Field(None, description="Embedding model configuration")
    vector_store: Optional[VectorStoreConfig] = Field(None, description="Vector store configuration")
    retrieval: RetrievalConfig = Field(..., description="Retrieval strategy configuration")
    document_processor: Optional[DocumentProcessorConfig] = Field(None, description="Document processor configuration")
    evaluation: Optional[EvaluationConfig] = Field(None, description="Evaluation configuration")
    
    # Pipeline name for identification
    name: str = Field("default", description="Name of the RAG pipeline")
    description: Optional[str] = Field(None, description="Description of the pipeline")
    
    @validator('embedding', 'vector_store', always=True)
    def validate_components(cls, v, values, **kwargs):
        field_name = kwargs.get('field_name')
        retrieval_strategy = values.get('retrieval', None)
        
        if retrieval_strategy is None:
            return v
            
        strategy_type = retrieval_strategy.strategy_type
        
        # Embedding-based retrieval needs embedding and vector_store
        if strategy_type == RetrievalStrategyType.EMBEDDING:
            if field_name == 'embedding' and v is None:
                raise ValueError("Embedding configuration is required for embedding-based retrieval")
            if field_name == 'vector_store' and v is None:
                raise ValueError("Vector store configuration is required for embedding-based retrieval")
        
        # TFIDF-based retrieval doesn't need embedding or vector_store
        # So no validation needed for TFIDF
        
        # Hybrid retrieval needs both embedding, vector_store, and TFIDF components
        if strategy_type == RetrievalStrategyType.HYBRID:
            if field_name == 'embedding' and v is None:
                raise ValueError("Embedding configuration is required for hybrid retrieval")
            if field_name == 'vector_store' and v is None:
                raise ValueError("Vector store configuration is required for hybrid retrieval")
        
        return v 
    
# Representa um modelo de embedding único
class EmbeddingConfig(BaseModel):
    model_type: str
    model_name: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = {}
    models: Optional[List["EmbeddingConfig"]] = None  # para modelo combinado
    weights: Optional[List[float]] = None

    @validator("models", always=True)
    def validate_combined_models(cls, v, values):
        if values.get("model_type") == "combined":
            if not v:
                raise ValueError("Combined model type requires 'models' field")
        return v

EmbeddingConfig.update_forward_refs()

# Representa múltiplos modelos com pesos
class CombinedEmbeddingConfig(BaseModel):
    models: List[EmbeddingConfig]
    weights: Optional[List[float]] = None

# Config principal do pipeline
class RAGConfig(BaseModel):
    embedding: Union[EmbeddingConfig, CombinedEmbeddingConfig]
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    evaluation: Optional[EvaluationConfig] = None