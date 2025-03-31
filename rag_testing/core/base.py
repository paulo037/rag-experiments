from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Base document class for storing text and metadata."""
    content: str = Field(..., description="The document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    id: Optional[str] = Field(None, description="Document ID")


class SearchResult(BaseModel):
    """Class to store search results with document and score information."""
    document: Document = Field(..., description="The retrieved document")
    score: float = Field(..., description="The relevance score")
    embedding_score: Optional[float] = Field(None, description="Embedding similarity score (for hybrid retrieval)")
    tfidf_score: Optional[float] = Field(None, description="TF-IDF similarity score (for hybrid retrieval)")

    @classmethod
    def from_document(cls, doc: Document, score_field: str = "distance") -> "SearchResult":
        """
        Create a SearchResult from a Document with score in metadata.
        
        Args:
            doc: Document with score in metadata
            score_field: Name of the metadata field containing the score
            
        Returns:
            SearchResult instance
        """
        # For embedding retrieve where lower distance is better
        if score_field == "distance":
            # Distance to similarity conversion
            score = 1.0 - doc.metadata.get(score_field, 0.0)
        else:
            # For TF-IDF where higher similarity is better
            score = doc.metadata.get(score_field, 0.0)
            
        # Check for hybrid retrieval fields
        embedding_score = doc.metadata.get("embedding_score", None)
        tfidf_score = doc.metadata.get("tfidf_score", None)
        combined_score = doc.metadata.get("combined_score", None)
        
        # Use combined score if available, otherwise use the converted score
        final_score = combined_score if combined_score is not None else score
        
        return cls(
            document=doc,
            score=final_score,
            embedding_score=embedding_score,
            tfidf_score=tfidf_score
        )


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
    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of search results with documents and scores
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
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents if a document processor is available.
        
        Args:
            documents: Documents to process
            
        Returns:
            Processed documents
        """
        return documents
    
    def ingest_documents(self, documents: List[Document]) -> None:
        """
        Ingest documents into the pipeline.
        Alias for index() method.
        
        Args:
            documents: Documents to ingest
        """
        self.index(documents) 