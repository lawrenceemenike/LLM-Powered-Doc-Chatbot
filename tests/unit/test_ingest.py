import pytest
from ingest.cleaner import TextCleaner
from ingest.splitter import TextSplitter

def test_text_cleaner():
    cleaner = TextCleaner()
    raw_text = "Hello   World! \u0000"
    cleaned = cleaner.clean(raw_text)
    assert cleaned == "Hello World!"

def test_text_splitter():
    splitter = TextSplitter(chunk_size=10, chunk_overlap=0)
    documents = [{"text": "Hello World", "source": "test", "page_number": 1}]
    chunks = splitter.split(documents)
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Hello"
    assert chunks[1]["text"] == "World"
