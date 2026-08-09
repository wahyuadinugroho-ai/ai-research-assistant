from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
import os
import tempfile
import nltk

# Notebook 4: NLTKTextSplitter requires 'punkt_tab' or 'punkt' depending on nltk version
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        nltk.download('punkt', quiet=True)

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
    Split documents into chunks for RAG using NLTK.
    """
    splitter = NLTKTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def load_and_split_papers(files):
    """
    Load multiple PDFs and split them into chunks.
    """
    documents = []

    for file in files:
        documents.extend(load_pdf(file))

    return documents, split_documents(documents)
