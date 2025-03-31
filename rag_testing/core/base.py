from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Base document class for storing text and metadata."""
    content: str = Field(..., description="The document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    id: Optional[str] = Field(None, description="Document ID")


class DocumentLoader(ABC):
    """Interface for document loading components."""
    
    @abstractmethod
    def load(self, source: Union[str, List[str]]) -> List[Document]:
        """
        Load documents from a source.
        
        Args:
            source: Path to a file, directory, or list of paths
            
        Returns:
            List of Document objects
        """
        pass


class DocumentProcessor(ABC):
    """Interface for document processing components."""
    
    @abstractmethod
    def process(self, documents: List[Document]) -> List[Document]:
        """
        Process a list of documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Processed documents
        """
        pass


class EmbeddingModel(ABC):
    """Interface for embedding models."""
    
    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of document embeddings
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding
        """
        pass


class VectorStore(ABC):
    """Interface for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of documents to add
            embeddings: List of document embeddings
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        pass


class RetrievalStrategy(ABC):
    """Interface for retrieval strategies."""
    
    @abstractmethod
    def setup(self, documents: List[Document]) -> None:
        """
        Set up the retrieval strategy with documents.
        
        Args:
            documents: List of documents for retrieval
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents relevant to a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        pass


class EvaluationMetric(ABC):
    """Interface for evaluation metrics."""
    
    @abstractmethod
    def evaluate(self, retrieved_docs: List[Document], relevant_docs: List[Document]) -> float:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Relevant documents (ground truth)
            
        Returns:
            Evaluation score
        """
        pass


class RAGPipeline(ABC):
    """Interface for RAG pipelines."""
    
    @abstractmethod
    def index(self, documents: List[Document]) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of documents to index
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        pass
    
    @abstractmethod
    def evaluate(self, queries: List[str], relevant_docs: Dict[str, List[Document]]) -> Dict[str, float]:
        """
        Evaluate the pipeline on a set of queries.
        
        Args:
            queries: List of query strings
            relevant_docs: Dictionary mapping queries to relevant documents
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass 