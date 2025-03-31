# RAG Testing Framework

A modular framework for testing different Retrieval-Augmented Generation (RAG) approaches, including various embedding models, hybrid retrieval strategies with TF-IDF, and more.

## Features

- Modular architecture for easy component swapping
- Pydantic models for configuration and validation
- Support for various embedding models
- Chromadb integration for vector storage
- Hybrid retrieval strategies (embeddings + TF-IDF)
- Evaluation framework for comparing approaches

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See the examples directory for sample implementations.

```python
# Example usage
from rag_testing.examples.embedding_tfidf import run_embedding_tfidf_example

run_embedding_tfidf_example()
```

## Project Structure

- `core/`: Core interfaces and abstract classes
- `embeddings/`: Embedding model implementations
- `retrieval/`: Retrieval strategy implementations
- `data/`: Document loading and processing
- `evaluation/`: Metrics and evaluation tools
- `config/`: Pydantic configuration models
- `examples/`: Example implementations

## Core Components

At the heart of the framework is `core/base.py`, which defines the fundamental interfaces:

- `Document`: Represents a text document with content, metadata, and an ID
- `DocumentLoader`: Interface for loading documents from various sources
- `DocumentProcessor`: Interface for processing documents (cleaning, splitting, etc.)
- `EmbeddingModel`: Interface for models that generate vector embeddings
- `VectorStore`: Interface for storing and retrieving embeddings
- `RetrievalStrategy`: Interface for different retrieval methods
- `EvaluationMetric`: Interface for retrieval quality metrics
- `RAGPipeline`: Interface for the complete RAG pipeline

These abstract classes ensure a consistent interface across implementations and make it easy to swap components.

## Data Processing

The `data/` module contains:

- `loaders.py`: Implementations for loading documents from text files, JSON, and CSV
- `processors.py`: Implementations for processing documents, including:
  - `TextSplitter`: Chunks documents into smaller pieces with overlap
  - `TextCleaner`: Cleans text by removing unwanted content
  - `MetadataProcessor`: Processes document metadata
  - `CompositeProcessor`: Combines multiple processors into a sequence

## Embedding Models

The `embeddings/models.py` file implements:

- `SentenceTransformerEmbedding`: Uses the SentenceTransformers library
- `CustomEmbedding`: Allows using custom embedding functions
- `MockEmbedding`: Creates random embeddings for testing

There's also a factory function `get_embedding_model()` that creates embedding models from configurations.

## Vector Stores

The `retrieval/vector_stores.py` file implements:

- `ChromaVectorStore`: Integration with the Chroma vector database
- Factory function for creating vector stores from configurations

## Retrieval Strategies

The `retrieval/strategies.py` file implements different approaches:

- `EmbeddingRetrieval`: Pure embedding-based similarity search
- `TfidfRetrieval`: Traditional TF-IDF retrieval
- `HybridRetrieval`: Combines embedding-based and TF-IDF scores

## Evaluation Metrics

The `evaluation/metrics.py` file implements:

- `PrecisionMetric`: Measures precision of retrieved documents
- `RecallMetric`: Measures recall of retrieved documents
- `F1Metric`: Calculates the F1 score
- `NDCGMetric`: Normalized Discounted Cumulative Gain
- `MRRMetric`: Mean Reciprocal Rank

## Pipeline Implementation

The `core/pipeline.py` file implements:

- `SimpleRAGPipeline`: Integrates all components into a complete pipeline
- `create_pipeline_from_config()`: Factory function to create pipelines from configurations

## Configuration System

The `config/models.py` file uses Pydantic for:

- Type-safe configuration with validation
- Enums for different component types
- Configuration for the complete RAG pipeline

## Demo and Example

The example implementation in `examples/embedding_tfidf.py` demonstrates:

- Creating a hybrid retrieval pipeline using embeddings and TF-IDF
- Working with sample documents
- Running retrieval queries
- Benchmarking performance

The `demo.py` script provides a command-line interface to:
- Try different retrieval strategies
- Adjust weights for hybrid retrieval
- Test with custom queries
- View detailed results with scores

## Key Strengths of the Framework

1. **Modularity**: Easy to swap components and test different configurations
2. **Type Safety**: Pydantic ensures valid configurations
3. **Extensibility**: New embedding models, retrieval strategies, or evaluation metrics can be added
4. **Configurability**: Detailed configuration options with sensible defaults
5. **Evaluation**: Built-in metrics to measure retrieval quality
6. **Benchmarking**: Tools to measure performance

You can use this framework to test various combinations of embedding models and retrieval strategies, helping you find the optimal approach for your specific RAG use case. 