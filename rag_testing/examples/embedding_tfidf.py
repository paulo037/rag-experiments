"""
Example implementation of a hybrid retrieval strategy using embeddings and TF-IDF.
"""

import os
import json
from typing import List

from rag_testing.core.base import Document
from rag_testing.pipelines.pipeline import SimpleRAGPipeline, create_pipeline_from_config
from rag_testing.config.models import (
    RAGConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    EvaluationConfig,
    EmbeddingModelType,
    VectorStoreType,
    RetrievalStrategyType
)
from rag_testing.data.processors import TextSplitter, TextCleaner, CompositeProcessor
from rag_testing.embeddings.models import SentenceTransformerEmbedding
from rag_testing.retrieval.vector_stores import ChromaVectorStore
from rag_testing.retrieval.strategies import HybridRetrieval
from rag_testing.evaluation.metrics import PrecisionMetric, RecallMetric, F1Metric


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    
    documents = [
        Document(
            content="Neural networks are a class of machine learning algorithms inspired by the human brain.",
            metadata={"category": "machine_learning", "source": "sample"},
            id="doc1"
        ),
        Document(
            content="Transformers have revolutionized natural language processing with attention mechanisms.",
            metadata={"category": "nlp", "source": "sample"},
            id="doc2"
        ),
        Document(
            content="Retrieval-Augmented Generation (RAG) combines retrieval systems with text generation.",
            metadata={"category": "rag", "source": "sample"},
            id="doc3"
        ),
        Document(
            content="TF-IDF stands for Term Frequency-Inverse Document Frequency, a numerical statistic used in information retrieval.",
            metadata={"category": "information_retrieval", "source": "sample"},
            id="doc4"
        ),
        Document(
            content="Vector search uses embeddings to find semantically similar documents in a vector space.",
            metadata={"category": "vector_search", "source": "sample"},
            id="doc5"
        ),
        Document(
            content="Embedding models convert text into numerical vectors that represent meaning.",
            metadata={"category": "embeddings", "source": "sample"},
            id="doc6"
        ),
        Document(
            content="Python is a popular programming language for machine learning and data science.",
            metadata={"category": "programming", "source": "sample"},
            id="doc7"
        ),
        Document(
            content="Large Language Models (LLMs) are trained on vast amounts of text data.",
            metadata={"category": "llm", "source": "sample"},
            id="doc8"
        ),
        Document(
            content="Sentence transformers are models specifically designed to create embeddings for sentences and paragraphs.",
            metadata={"category": "embeddings", "source": "sample"},
            id="doc9"
        ),
        Document(
            content="Chroma is a vector database designed for building AI applications with embeddings.",
            metadata={"category": "vector_db", "source": "sample"},
            id="doc10"
        )
    ]
    
    return documents


def save_sample_documents(documents: List[Document], output_dir: str = "data"):
    """Save sample documents to disk."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    with open(os.path.join(output_dir, "sample_docs.json"), "w") as f:
        json_data = [
            {
                "id": doc.id,
                "content": doc.content,
                **doc.metadata
            }
            for doc in documents
        ]
        json.dump(json_data, f, indent=2)
    
    # Save as text files
    text_dir = os.path.join(output_dir, "texts")
    os.makedirs(text_dir, exist_ok=True)
    
    for doc in documents:
        with open(os.path.join(text_dir, f"{doc.id}.txt"), "w") as f:
            f.write(doc.content)


def create_config() -> RAGConfig:
    """Create a configuration for the hybrid retrieval pipeline."""
    
    config = RAGConfig(
        name="embedding_tfidf_example",
        description="Example pipeline using hybrid retrieval with embeddings and TF-IDF",
        
        # Embedding model configuration
        embedding=EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name="all-MiniLM-L6-v2",
            model_kwargs={}
        ),
        
        # Vector store configuration
        vector_store=VectorStoreConfig(
            store_type=VectorStoreType.CHROMA,
            collection_name="hybrid_example",
            persist_directory="data/chroma_db"
        ),
        
        # Retrieval strategy configuration
        retrieval=RetrievalConfig(
            strategy_type=RetrievalStrategyType.HYBRID,
            top_k=5,
            embedding_weight=0.7,
            tfidf_weight=0.3
        ),
        
        # Evaluation configuration
        evaluation=EvaluationConfig(
            metrics=["precision", "recall", "f1", "ndcg@5"]
        )
    )
    
    return config


def run_embedding_tfidf_example():
    """Run the embedding + TF-IDF example."""
    
    print("Creating sample documents...")
    documents = create_sample_documents()
    
    # Save sample documents
    save_sample_documents(documents)
    
    print("Creating RAG pipeline...")
    config = create_config()
    pipeline = create_pipeline_from_config(config)
    
    print("Indexing documents...")
    pipeline.index(documents)
    
    print("\nTesting queries:")
    test_queries = [
        "What are neural networks?",
        "How do transformers work?",
        "Tell me about embedding models",
        "What is RAG?",
        "Vector search methods"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = pipeline.retrieve(query, k=3)
        
        for i, doc in enumerate(results):
            # Extract scores from metadata
            embedding_score = doc.metadata.get("embedding_score", 0)
            tfidf_score = doc.metadata.get("tfidf_score", 0)
            combined_score = doc.metadata.get("combined_score", 0)
            
            print(f"Result {i+1}: [ID: {doc.id}] (Emb: {embedding_score:.4f}, TFIDF: {tfidf_score:.4f}, Combined: {combined_score:.4f})")
            print(f"  {doc.content}")
    
    # Benchmark performance
    print("\nBenchmarking retrieval performance...")
    benchmark_results = pipeline.benchmark(test_queries)
    
    print("\nBenchmark Results:")
    print(f"  Average retrieval time: {benchmark_results['avg_retrieval_time']*1000:.2f} ms")
    print(f"  Number of documents: {benchmark_results['num_documents']}")
    print(f"  Index time: {benchmark_results['index_time']:.2f} s")
    
    print("\nDone!")


def create_manual_hybrid_pipeline():
    """Create a hybrid pipeline manually without using the config-based factory."""
    
    # Create embedding model
    embedding_model = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vector_store = ChromaVectorStore(
        collection_name="manual_hybrid",
        persist_directory="data/chroma_manual"
    )
    
    # Create hybrid retrieval strategy
    retrieval_strategy = HybridRetrieval(
        embedding_model=embedding_model,
        vector_store=vector_store,
        embedding_weight=0.7,
        tfidf_weight=0.3
    )
    
    # Create document processor
    document_processor = CompositeProcessor([
        TextCleaner(remove_extra_whitespace=True),
        TextSplitter(chunk_size=1000, chunk_overlap=200)
    ])
    
    # Create evaluation metrics
    evaluation_metrics = [
        PrecisionMetric(),
        RecallMetric(),
        F1Metric()
    ]
    
    # Create pipeline
    pipeline = SimpleRAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        retrieval_strategy=retrieval_strategy,
        document_processor=document_processor,
        evaluation_metrics=evaluation_metrics
    )
    
    return pipeline


if __name__ == "__main__":
    run_embedding_tfidf_example() 