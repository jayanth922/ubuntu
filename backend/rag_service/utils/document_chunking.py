import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional

class DocumentChunker:
    """Split documents into chunks for better retrieval"""
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 128,
        separator: str = "\n"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def _generate_chunk_id(self, text: str, doc_id: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{doc_id}_chunk_{chunk_index}_{content_hash}"
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a document into overlapping chunks"""
        doc_id = document.get("id", "unknown")
        content = document.get("content", "")
        
        # If content is too short, return the document as is
        if len(content) <= self.chunk_size:
            document["chunk_id"] = self._generate_chunk_id(content, doc_id, 0)
            return [document]
        
        # Split by separator if possible
        if self.separator and self.separator in content:
            segments = content.split(self.separator)
            chunks = []
            current_chunk = ""
            chunk_index = 0
            
            for segment in segments:
                # If adding this segment exceeds chunk size, save the current chunk and start a new one
                if len(current_chunk) + len(segment) + len(self.separator) > self.chunk_size:
                    if current_chunk:  # Only save if we have content
                        chunk_doc = document.copy()
                        chunk_doc["content"] = current_chunk.strip()
                        chunk_doc["chunk_id"] = self._generate_chunk_id(current_chunk, doc_id, chunk_index)
                        chunk_doc["is_chunk"] = True
                        chunk_doc["parent_id"] = doc_id
                        chunks.append(chunk_doc)
                        chunk_index += 1
                        
                        # Start new chunk with overlap
                        current_chunk = self._get_overlap_text(current_chunk)
                
                # Add the segment to the current chunk
                if current_chunk:
                    current_chunk += self.separator + segment
                else:
                    current_chunk = segment
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_doc = document.copy()
                chunk_doc["content"] = current_chunk.strip()
                chunk_doc["chunk_id"] = self._generate_chunk_id(current_chunk, doc_id, chunk_index)
                chunk_doc["is_chunk"] = True
                chunk_doc["parent_id"] = doc_id
                chunks.append(chunk_doc)
            
            return chunks
        
        # If no separator or not useful, fall back to character chunking
        chunks = []
        for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
            chunk_text = content[i:i + self.chunk_size]
            if chunk_text:  # Only add non-empty chunks
                chunk_doc = document.copy()
                chunk_doc["content"] = chunk_text
                chunk_doc["chunk_id"] = self._generate_chunk_id(chunk_text, doc_id, i // (self.chunk_size - self.chunk_overlap))
                chunk_doc["is_chunk"] = True
                chunk_doc["parent_id"] = doc_id
                chunks.append(chunk_doc)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Extract the overlap portion from the end of a text"""
        words = text.split()
        if len(words) <= self.chunk_overlap // 10:  # Approximate words in overlap
            return text
        
        return " ".join(words[-(self.chunk_overlap // 10):])
    
    def chunk_collection(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a collection of documents into chunks"""
        chunked_docs = []
        for document in documents:
            chunked_docs.extend(self.chunk_document(document))
        return chunked_docs