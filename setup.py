from setuptools import setup, find_packages

setup(
    name="rag_testing",
    version="0.1.0",
    description="A framework for testing RAG approaches",
    author="RAG Testing Team",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "chromadb>=0.4.22",
        "sentence-transformers>=2.2.2",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "langchain>=0.0.300",
        "pytest>=7.0.0",
        "tqdm>=4.66.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 