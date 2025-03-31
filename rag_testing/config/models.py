from pydantic import BaseModel, Field
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
    embedding: EmbeddingConfig = Field(..., description="Embedding model configuration")
    vector_store: VectorStoreConfig = Field(..., description="Vector store configuration")
    retrieval: RetrievalConfig = Field(..., description="Retrieval strategy configuration")
    document_processor: Optional[DocumentProcessorConfig] = Field(None, description="Document processor configuration")
    evaluation: Optional[EvaluationConfig] = Field(None, description="Evaluation configuration")
    
    # Pipeline name for identification
    name: str = Field("default", description="Name of the RAG pipeline")
    description: Optional[str] = Field(None, description="Description of the pipeline") 