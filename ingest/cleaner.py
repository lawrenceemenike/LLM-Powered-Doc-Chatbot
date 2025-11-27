import re
import unicodedata

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        """
        Cleans and normalizes text.
        
        Args:
            text: The input text string.
            
        Returns:
            The cleaned text string.
        """
        if not text:
            return ""
            
        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        
        return text.strip()
