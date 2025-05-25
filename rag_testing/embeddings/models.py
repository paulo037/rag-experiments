from typing import List, Any, Optional
import numpy as np

from rag_testing.core.base import Document, EmbeddingModel
from rag_testing.config.models import EmbeddingConfig


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        """
        Initialize the SentenceTransformer embedding model.
        
        Args:
            model_name: The name of the SentenceTransformer model
            **kwargs: Additional kwargs for the model
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, **kwargs)
        except ImportError:
            raise ImportError(
                "SentenceTransformers package not found. "
                "Please install it with `pip install sentence-transformers`."
            )
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of document embeddings
        """
        texts = [doc.content for doc in documents]
        embeddings = self.model.encode(texts)
        
        # Convert to native list type (from numpy array)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding
        """
        embedding = self.model.encode(query)
        
        # Convert to native list type (from numpy array)
        return embedding.tolist()


class CustomEmbedding(EmbeddingModel):
    """Custom embedding model with a provided embedding function."""
    
    def __init__(self, 
                 document_embedding_fn: callable, 
                 query_embedding_fn: Optional[callable] = None):
        """
        Initialize the custom embedding model.
        
        Args:
            document_embedding_fn: Function to embed documents
            query_embedding_fn: Function to embed queries (defaults to document_embedding_fn)
        """
        self.document_embedding_fn = document_embedding_fn
        self.query_embedding_fn = query_embedding_fn or document_embedding_fn
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Embed a list of documents using the provided function.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of document embeddings
        """
        # Apply the embedding function to each document
        return self.document_embedding_fn(documents)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string using the provided function.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding
        """
        # Apply the query embedding function
        return self.query_embedding_fn(query)


class MockEmbedding(EmbeddingModel):
    """Mock embedding model for testing."""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the mock embedding model.
        
        Args:
            embedding_dim: Dimension of the mock embeddings
        """
        self.embedding_dim = embedding_dim
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate mock embeddings for documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of mock document embeddings
        """
        # Generate random embeddings for testing
        np.random.seed(42)  # For reproducibility
        embeddings = []
        
        for doc in documents:
            # Make embedding somewhat content-dependent by using hash of content
            content_hash = hash(doc.content) % 10000
            np.random.seed(content_hash)
            
            embedding = np.random.randn(self.embedding_dim).tolist()
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate mock embedding for a query.
        
        Args:
            query: Query string to embed
            
        Returns:
            Mock query embedding
        """
        # Generate random embedding for testing
        np.random.seed(hash(query) % 10000)
        return np.random.randn(self.embedding_dim).tolist()


def get_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    model_type = config.model_type
    model_name = config.model_name
    model_kwargs = config.model_kwargs

    if model_type == "sentence_transformer":
        return SentenceTransformerEmbedding(model_name=model_name, **model_kwargs)
    elif model_type == "mock":
        embedding_dim = model_kwargs.get("embedding_dim", 384)
        return MockEmbedding(embedding_dim=embedding_dim)
    elif model_type == "combined":
        submodels = [
            get_embedding_model(sub_config)
            for sub_config in config.models
        ]
        weights = config.weights
        return CombinedEmbeddingModel(submodels, weights)
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")
    

class CombinedEmbeddingModel:
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)

    def embed_query(self, query: str) -> List[float]:
        embeddings = [model.embed_query(query) for model in self.models]
        weighted = [
            [e * w for e in emb]
            for emb, w in zip(embeddings, self.weights)
        ]
        return [sum(values) / len(self.models) for values in zip(*weighted)]

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        all_embeddings = [model.embed_documents(docs) for model in self.models]
        combined = []

        for i in range(len(docs)):
            # Para cada documento, combinar os embeddings dos modelos
            weighted_doc_embs = [
                [emb[i][d] * self.weights[m] for d in range(len(emb[i]))]
                for m, emb in enumerate(all_embeddings)
            ]
            combined_doc = [sum(values) / len(self.models) for values in zip(*weighted_doc_embs)]
            combined.append(combined_doc)

        return combined 