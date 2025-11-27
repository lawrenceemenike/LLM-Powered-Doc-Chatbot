import pdfplumber
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self):
        pass

    def parse(self, file_path: str) -> List[Dict]:
        """
        Parses a PDF file and extracts text page by page.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            A list of dictionaries, where each dictionary represents a page
            and contains 'page_number' and 'text'.
        """
        documents = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        documents.append({
                            "page_number": i + 1,
                            "text": text,
                            "source": file_path
                        })
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise e
            
        return documents
