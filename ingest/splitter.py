from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, documents: List[Dict]) -> List[Dict]:
        """
        Splits documents into chunks.
        
        Args:
            documents: List of dictionaries with 'text' and metadata.
            
        Returns:
            List of dictionaries with 'text', 'chunk_id', and metadata.
        """
        chunked_docs = []
        for doc in documents:
            text = doc.get("text", "")
            chunks = self.splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.copy()
                chunk_metadata["text"] = chunk
                chunk_metadata["chunk_id"] = f"{doc.get('source', 'unknown')}_p{doc.get('page_number', 0)}_c{i}"
                chunked_docs.append(chunk_metadata)
                
        return chunked_docs
