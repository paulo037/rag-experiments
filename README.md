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