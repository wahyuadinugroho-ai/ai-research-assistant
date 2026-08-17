from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.prompts import SUGGESTION_PROMPT
import re


def create_chroma_vectorstore(documents, embeddings):
    """Create a Chroma vector store."""
    return Chroma.from_documents(documents, embeddings)


def create_faiss_vectorstore(documents, embeddings):
    """Create a FAISS vector store."""
    return FAISS.from_documents(documents, embeddings)


def quick_action_stream(documents, llm, prompt_template):
    """Generate a structured response for quick actions using LCEL and stream response.

    Uses a generous character limit of 120,000 chars (~25k tokens) to analyze full papers
    and multi-paper literature reviews comprehensively with Gemini.
    """
    if not documents:

        def _empty_gen():
            yield "Tidak ada dokumen yang dipilih atau dokumen tidak memiliki konten teks."

        return _empty_gen()

    MAX_CHARS = 120_000
    raw_text = "\n\n".join(
        f"--- Dokumen: {doc.metadata.get('filename', 'Unknown')} (Halaman {doc.metadata.get('page', 0) + 1}) ---\n{doc.page_content}"
        for doc in documents
        if doc.page_content and doc.page_content.strip()
    )

    text = raw_text[:MAX_CHARS]
    if len(raw_text) > MAX_CHARS:
        text += "\n\n[Catatan: Sebagian teks dipotong karena melebihi batas 120.000 karakter.]"

    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({"text": text})


def generate_suggestions(documents, llm):
    """Generate 3 dynamic questions based on the document."""
    default_questions = [
        "Apa kontribusi utamanya?",
        "Bandingkan metodenya",
        "Ringkas temuan akhirnya",
    ]

    if not documents:
        return default_questions

    # Take up to 2500 chars for richer context while remaining fast
    valid_contents = [
        d.page_content.strip()
        for d in documents
        if d.page_content and d.page_content.strip()
    ]
    if not valid_contents:
        return default_questions

    text = valid_contents[0][:2500]
    prompt = PromptTemplate.from_template(SUGGESTION_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        res = chain.invoke({"text": text})
        # Clean string from numbering or bullet points
        questions = [
            re.sub(r"^[\d\.\-\*\s]+", "", q).strip()
            for q in res.split("\n")
            if q.strip()
        ]
        questions = [q for q in questions if len(q) > 5][:3]

        # Fallback if not enough questions parsed
        while len(questions) < 3:
            questions.append(default_questions[len(questions)])

        return questions
    except Exception:
        return default_questions
