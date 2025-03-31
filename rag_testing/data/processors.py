import re
from typing import List, Dict, Any, Optional, Callable
import uuid

from rag_testing.core.base import Document, DocumentProcessor


class TextSplitter(DocumentProcessor):
    """Split documents into chunks based on size."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: The target size of each document chunk
            chunk_overlap: The overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of chunked documents
        """
        chunked_documents = []
        
        for doc in documents:
            content = doc.content
            
            # Skip empty documents
            if not content.strip():
                continue
            
            # Simple character-level chunking
            start = 0
            chunks = []
            
            while start < len(content):
                end = min(start + self.chunk_size, len(content))
                
                # If we're not at the beginning or end, try to find a sensible boundary
                if start > 0 and end < len(content):
                    # Look for a newline or period to break on
                    last_newline = content.rfind('\n', start, end)
                    last_period = content.rfind('. ', start, end)
                    
                    if last_newline > start + self.chunk_size // 2:
                        end = last_newline + 1
                    elif last_period > start + self.chunk_size // 2:
                        end = last_period + 2
                
                chunk_content = content[start:end]
                chunk_id = f"{doc.id}_{len(chunks)}" if doc.id else str(uuid.uuid4())
                
                # Create a new document for the chunk
                chunk_doc = Document(
                    content=chunk_content,
                    metadata={
                        **doc.metadata,
                        "chunk_index": len(chunks),
                        "parent_id": doc.id or "unknown"
                    },
                    id=chunk_id
                )
                
                chunks.append(chunk_doc)
                
                # Move to next chunk with overlap
                start = end - self.chunk_overlap
            
            chunked_documents.extend(chunks)
        
        return chunked_documents


class TextCleaner(DocumentProcessor):
    """Clean text by removing extra whitespace, special characters, etc."""
    
    def __init__(self, 
                 remove_extra_whitespace: bool = True,
                 lowercase: bool = False,
                 remove_urls: bool = False,
                 remove_emails: bool = False,
                 custom_replacements: Dict[str, str] = None):
        """
        Initialize the text cleaner.
        
        Args:
            remove_extra_whitespace: Whether to remove extra whitespace
            lowercase: Whether to convert text to lowercase
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            custom_replacements: Dictionary of custom text replacements
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.custom_replacements = custom_replacements or {}
    
    def process(self, documents: List[Document]) -> List[Document]:
        """
        Clean document text.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of cleaned documents
        """
        cleaned_documents = []
        
        for doc in documents:
            content = doc.content
            
            # Apply text cleaning operations
            if self.remove_urls:
                content = re.sub(r'https?://\S+', '', content)
            
            if self.remove_emails:
                content = re.sub(r'\S+@\S+', '', content)
            
            if self.remove_extra_whitespace:
                content = re.sub(r'\s+', ' ', content).strip()
            
            if self.lowercase:
                content = content.lower()
            
            # Apply custom replacements
            for old_text, new_text in self.custom_replacements.items():
                content = content.replace(old_text, new_text)
            
            # Create a new document with cleaned content
            cleaned_doc = Document(
                content=content,
                metadata=doc.metadata,
                id=doc.id
            )
            
            cleaned_documents.append(cleaned_doc)
        
        return cleaned_documents


class MetadataProcessor(DocumentProcessor):
    """Process document metadata."""
    
    def __init__(self, metadata_fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Initialize the metadata processor.
        
        Args:
            metadata_fn: Function to process metadata
        """
        self.metadata_fn = metadata_fn
    
    def process(self, documents: List[Document]) -> List[Document]:
        """
        Process document metadata.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of documents with processed metadata
        """
        processed_documents = []
        
        for doc in documents:
            processed_metadata = self.metadata_fn(doc.metadata)
            
            processed_doc = Document(
                content=doc.content,
                metadata=processed_metadata,
                id=doc.id
            )
            
            processed_documents.append(processed_doc)
        
        return processed_documents


class CompositeProcessor(DocumentProcessor):
    """Apply multiple document processors in sequence."""
    
    def __init__(self, processors: List[DocumentProcessor]):
        """
        Initialize the composite processor.
        
        Args:
            processors: List of document processors to apply
        """
        self.processors = processors
    
    def process(self, documents: List[Document]) -> List[Document]:
        """
        Apply multiple processors in sequence.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed documents
        """
        result = documents
        
        for processor in self.processors:
            result = processor.process(result)
        
        return result 