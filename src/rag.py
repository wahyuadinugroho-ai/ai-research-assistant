from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import TOP_K
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
    
    Caps input text at 15,000 chars to avoid token overflow on large PDFs.
    """
    MAX_CHARS = 15_000
    raw_text = "\n\n".join(document.page_content for document in documents)
    text = raw_text[:MAX_CHARS]
    if len(raw_text) > MAX_CHARS:
        text += "\n\n[Konten dipotong karena dokumen terlalu panjang. Ringkasan berdasarkan bagian pertama dokumen.]"
    prompt = PromptTemplate.from_template(prompt_template)
    
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    stream = chain.stream({"text": text})
                
    return stream

def generate_suggestions(documents, llm):
    """Generate 3 dynamic questions based on the document."""
    if not documents:
        return ["Apa kontribusi utamanya?", "Bandingkan metodenya", "Ringkas temuan akhirnya"]
    
    # Ambil 1500 karakter pertama saja agar cepat dan hemat token
    text = documents[0].page_content[:1500]
    prompt = PromptTemplate.from_template(SUGGESTION_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        res = chain.invoke({"text": text})
        # Clean string from numbering or bullet points
        questions = [re.sub(r'^[\d\.\-\*\s]+', '', q).strip() for q in res.split("\n") if q.strip()]
        questions = [q for q in questions if len(q) > 5][:3]
        
        # Fallback if not enough questions parsed
        while len(questions) < 3:
            questions.append("Jelaskan lebih lanjut!")
            
        return questions
    except Exception:
        return ["Apa kontribusi utamanya?", "Bandingkan metodenya", "Ringkas temuan akhirnya"]
