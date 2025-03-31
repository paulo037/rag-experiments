import os
import glob
import json
import csv
from typing import List, Union, Dict, Any
import pandas as pd

from rag_testing.core.base import Document, DocumentLoader


class TextFileLoader(DocumentLoader):
    """Loads documents from text files."""
    
    def load(self, source: Union[str, List[str]]) -> List[Document]:
        """
        Load documents from text files.
        
        Args:
            source: Path to a file, directory, or list of paths
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if isinstance(source, str):
            if os.path.isdir(source):
                # If source is a directory, get all text files
                source = glob.glob(os.path.join(source, "*.txt"))
            else:
                # If source is a file, wrap it in a list
                source = [source]
        
        for file_path in source:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                metadata = {
                    "source": file_path,
                    "filename": os.path.basename(file_path)
                }
                
                documents.append(Document(
                    content=content,
                    metadata=metadata,
                    id=os.path.basename(file_path)
                ))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents


class JsonLoader(DocumentLoader):
    """Loads documents from JSON files."""
    
    def __init__(self, content_key: str = "content", metadata_keys: List[str] = None):
        """
        Initialize the JSON loader.
        
        Args:
            content_key: Key for document content in JSON
            metadata_keys: Keys for metadata in JSON
        """
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []
    
    def load(self, source: Union[str, List[str]]) -> List[Document]:
        """
        Load documents from JSON files.
        
        Args:
            source: Path to a file, directory, or list of paths
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if isinstance(source, str):
            if os.path.isdir(source):
                # If source is a directory, get all JSON files
                source = glob.glob(os.path.join(source, "*.json"))
            else:
                # If source is a file, wrap it in a list
                source = [source]
        
        for file_path in source:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle both single document and list of documents
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
                
                for i, item in enumerate(items):
                    if self.content_key not in item:
                        continue
                    
                    content = item[self.content_key]
                    
                    # Extract metadata
                    metadata = {
                        "source": file_path,
                        "filename": os.path.basename(file_path)
                    }
                    
                    for key in self.metadata_keys:
                        if key in item:
                            metadata[key] = item[key]
                    
                    doc_id = f"{os.path.basename(file_path)}_{i}"
                    if "id" in item:
                        doc_id = item["id"]
                    
                    documents.append(Document(
                        content=content,
                        metadata=metadata,
                        id=doc_id
                    ))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents


class CSVLoader(DocumentLoader):
    """Loads documents from CSV files."""
    
    def __init__(self, content_column: str, metadata_columns: List[str] = None):
        """
        Initialize the CSV loader.
        
        Args:
            content_column: Column name for document content
            metadata_columns: Column names for metadata
        """
        self.content_column = content_column
        self.metadata_columns = metadata_columns or []
    
    def load(self, source: Union[str, List[str]]) -> List[Document]:
        """
        Load documents from CSV files.
        
        Args:
            source: Path to a file, directory, or list of paths
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if isinstance(source, str):
            if os.path.isdir(source):
                # If source is a directory, get all CSV files
                source = glob.glob(os.path.join(source, "*.csv"))
            else:
                # If source is a file, wrap it in a list
                source = [source]
        
        for file_path in source:
            try:
                df = pd.read_csv(file_path)
                
                if self.content_column not in df.columns:
                    print(f"Content column '{self.content_column}' not found in {file_path}")
                    continue
                
                for i, row in df.iterrows():
                    content = str(row[self.content_column])
                    
                    # Extract metadata
                    metadata = {
                        "source": file_path,
                        "filename": os.path.basename(file_path),
                        "row_idx": i
                    }
                    
                    for col in self.metadata_columns:
                        if col in df.columns:
                            metadata[col] = row[col]
                    
                    doc_id = f"{os.path.basename(file_path)}_{i}"
                    if "id" in df.columns:
                        doc_id = str(row["id"])
                    
                    documents.append(Document(
                        content=content,
                        metadata=metadata,
                        id=doc_id
                    ))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents 