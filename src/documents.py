from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import NLTKTextSplitter, RecursiveCharacterTextSplitter
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
import os
import tempfile
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass


def load_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()

        for document in documents:
            document.metadata["filename"] = uploaded_file.name

        return documents
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def split_documents(documents):
    """
    Split documents into chunks for RAG using NLTK (with RecursiveCharacterTextSplitter fallback).
    """
    if not documents:
        return []

    try:
        splitter = NLTKTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
    except Exception:
        # Fallback to standard recursive text splitter if NLTK fails
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

    # Filter out empty or whitespace-only chunks
    return [c for c in chunks if c.page_content and c.page_content.strip()]


def load_and_split_papers(files):
    """
    Load multiple PDFs and split them into chunks.
    Validates that extractable text exists.
    """
    documents = []

    for file in files:
        docs = load_pdf(file)
        documents.extend(docs)

    # Check if there is actual readable text
    total_text_length = sum(len(doc.page_content.strip()) for doc in documents)
    if total_text_length < 20:
        raise ValueError(
            "PDF yang diunggah tidak mengandung teks yang dapat dibaca. "
            "Pastikan PDF bukan hasil scan gambar murni tanpa OCR dan tidak terenkripsi."
        )

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError(
            "Gagal memecah teks PDF menjadi potongan chunk untuk embedding."
        )

    return documents, chunks
