from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_testing.core.base import Document, EmbeddingModel, VectorStore, RetrievalStrategy
from rag_testing.config.models import RetrievalConfig


class EmbeddingRetrieval(RetrievalStrategy):
    """Retrieval strategy using embeddings."""
    
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        """
        Initialize the embedding retrieval strategy.
        
        Args:
            embedding_model: Model to generate embeddings
            vector_store: Vector store for similarity search
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def setup(self, documents: List[Document]) -> None:
        """
        Set up the retrieval strategy with documents.
        
        Args:
            documents: List of documents for retrieval
        """
        # Generate embeddings for documents
        embeddings = self.embedding_model.embed_documents(documents)
        
        # Add documents and embeddings to the vector store
        self.vector_store.add_documents(documents, embeddings)
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents relevant to a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search for similar documents
        return self.vector_store.search(query_embedding, k=k)


class TfidfRetrieval(RetrievalStrategy):
    """Retrieval strategy using TF-IDF."""
    
    def __init__(self):
        """Initialize the TF-IDF retrieval strategy."""
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.tfidf_matrix = None
        self.documents = []
    
    def setup(self, documents: List[Document]) -> None:
        """
        Set up the retrieval strategy with documents.
        
        Args:
            documents: List of documents for retrieval
        """
        self.documents = documents
        
        # Extract document content
        contents = [doc.content for doc in documents]
        
        # Fit and transform the documents
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents relevant to a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Transform the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k document indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Retrieve documents and add similarity score to metadata
        retrieved_docs = []
        for idx in top_indices:
            doc = self.documents[idx]
            
            # Make a copy of the document with similarity score in metadata
            metadata = {**doc.metadata, "similarity": float(similarities[idx])}
            retrieved_doc = Document(
                content=doc.content,
                metadata=metadata,
                id=doc.id
            )
            
            retrieved_docs.append(retrieved_doc)
        
        return retrieved_docs


class HybridRetrieval(RetrievalStrategy):
    """Hybrid retrieval strategy combining embeddings and TF-IDF."""

    def __init__(self, 
                 embedding_model: EmbeddingModel, 
                 vector_store: VectorStore,
                 embedding_weight: float = 0.5,
                 tfidf_weight: float = 0.5,
                 fusion_method: str = "sum"):  # soma ponderada
        """
        Initialize the hybrid retrieval strategy.

        Args:
            embedding_model: Model to generate embeddings
            vector_store: Vector store for similarity search
            embedding_weight: Weight for embedding scores
            tfidf_weight: Weight for TF-IDF scores
            fusion_method: Method for fusing scores: 'sum', 'mean', or 'max'
        """
        self.embedding_retrieval = EmbeddingRetrieval(embedding_model, vector_store)
        self.tfidf_retrieval = TfidfRetrieval()
        self.embedding_weight = embedding_weight
        self.tfidf_weight = tfidf_weight
        self.fusion_method = fusion_method.lower()
        self.documents = []

    def setup(self, documents: List[Document]) -> None:
        self.documents = documents
        self.embedding_retrieval.setup(documents)
        self.tfidf_retrieval.setup(documents)

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        embedding_docs = self.embedding_retrieval.retrieve(query, k=k * 2)
        tfidf_docs = self.tfidf_retrieval.retrieve(query, k=k * 2)

        doc_scores = {}

        for doc in embedding_docs:
            doc_id = doc.id
            embedding_score = 1.0 - doc.metadata.get("distance", 0.0)
            doc_scores[doc_id] = {
                "doc": doc,
                "embedding_score": embedding_score * self.embedding_weight,
                "tfidf_score": 0.0,
            }

        for doc in tfidf_docs:
            doc_id = doc.id
            tfidf_score = doc.metadata.get("similarity", 0.0)
            if doc_id in doc_scores:
                doc_scores[doc_id]["tfidf_score"] = tfidf_score * self.tfidf_weight
            else:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "embedding_score": 0.0,
                    "tfidf_score": tfidf_score * self.tfidf_weight,
                }

        # fusion method
        for doc_id, entry in doc_scores.items():
            if self.fusion_method == "sum":
                entry["combined_score"] = entry["embedding_score"] + entry["tfidf_score"]
            elif self.fusion_method == "mean":
                entry["combined_score"] = (entry["embedding_score"] + entry["tfidf_score"]) / 2
            elif self.fusion_method == "max":
                entry["combined_score"] = max(entry["embedding_score"], entry["tfidf_score"])
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:k]

        retrieved_docs = []
        for entry in sorted_docs:
            doc = entry["doc"]
            metadata = {
                **doc.metadata,
                "embedding_score": entry["embedding_score"],
                "tfidf_score": entry["tfidf_score"],
                "combined_score": entry["combined_score"]
            }
            retrieved_docs.append(Document(
                content=doc.content,
                metadata=metadata,
                id=doc.id
            ))

        return retrieved_docs


def get_retrieval_strategy(
    config: RetrievalConfig,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore
) -> RetrievalStrategy:
    """
    Factory function to get a retrieval strategy based on configuration.
    
    Args:
        config: Retrieval configuration
        embedding_model: Embedding model for vector retrieval
        vector_store: Vector store for similarity search
        
    Returns:
        Configured retrieval strategy
    """
    strategy_type = config.strategy_type
    
    if strategy_type == "embedding":
        return EmbeddingRetrieval(embedding_model, vector_store)
    elif strategy_type == "tfidf":
        return TfidfRetrieval()
    elif strategy_type == "hybrid":
        return HybridRetrieval(
            embedding_model, 
            vector_store,
            embedding_weight=config.embedding_weight,
            tfidf_weight=config.tfidf_weight,
            fusion_method=config.fusion_method  
    )
    else:
        raise ValueError(f"Unsupported retrieval strategy type: {strategy_type}") 