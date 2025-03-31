from typing import Dict, List, Any, Optional, Tuple
import os

from rag_testing.core.base import Document, VectorStore
from rag_testing.config.models import VectorStoreConfig


class ChromaVectorStore(VectorStore):
    """Vector store using Chroma DB."""
    
    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory: Optional[str] = None,
                 embedding_function = None,
                 **kwargs):
        """
        Initialize the Chroma vector store.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist Chroma data
            embedding_function: Embedding function for Chroma
            **kwargs: Additional arguments for Chroma client
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB package not found. "
                "Please install it with `pip install chromadb`."
            )
        
        # Initialize Chroma client
        client_settings = Settings(**kwargs)
        
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory, settings=client_settings)
        else:
            self.client = chromadb.Client(settings=client_settings)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}  # Default to cosine similarity
        )
        
        # Maps document IDs to documents
        self.documents = {}
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of documents to add
            embeddings: List of document embeddings
        """
        if not documents:
            return
        
        ids = [doc.id if doc.id else str(i) for i, doc in enumerate(documents)]
        documents_content = [doc.content for doc in documents]
        documents_metadata = [doc.metadata for doc in documents]
        
        # Add documents to Chroma
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents_content,
            metadatas=documents_metadata
        )
        
        # Store documents for later retrieval
        for i, doc in enumerate(documents):
            self.documents[ids[i]] = doc
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_docs = []
        
        if not results["ids"][0]:
            return []
        
        for i, doc_id in enumerate(results["ids"][0]):
            content = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Add distance to metadata
            metadata["distance"] = distance
            
            doc = Document(
                content=content,
                metadata=metadata,
                id=doc_id
            )
            
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def delete(self, document_ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
        """
        self.collection.delete(ids=document_ids)
        
        # Remove from local document store
        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
    
    def get_collection(self):
        """Get the underlying Chroma collection."""
        return self.collection


def get_vector_store(config: VectorStoreConfig) -> VectorStore:
    """
    Factory function to get a vector store based on configuration.
    
    Args:
        config: Vector store configuration
        
    Returns:
        Configured vector store
    """
    store_type = config.store_type
    
    if store_type == "chroma":
        return ChromaVectorStore(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            **config.store_kwargs
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}") 