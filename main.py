from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from src.agent import create_research_agent
from src.config import (
    GOOGLE_API_KEY as ENV_GEMINI_KEY, 
    EXA_API_KEY as ENV_EXA_KEY,
    MAX_HISTORY_TURNS
)
from src.documents import load_and_split_papers
from src.helpers import extract_text
from src.history import (
    list_sessions, list_archived_sessions, load_session, save_session,
    archive_session, unarchive_session, delete_session
)
from src.models import get_embeddings, get_llm
from src.prompts import (
    SUMMARY_PROMPT, CITATION_PROMPT, GLOSSARY_PROMPT, CRITIQUE_PROMPT,
    INSIGHT_PROMPT, GAP_PROMPT, METHODOLOGY_PROMPT, LITERATURE_REVIEW_PROMPT,
    METADATA_PROMPT, TITLE_PROMPT,
)
from src.rag import create_chroma_vectorstore, create_faiss_vectorstore, quick_action_stream, generate_suggestions
from typing import List, Optional
import base64
import json
import streamlit as st
import traceback
import uuid


class PaperMetadataModel(BaseModel):
    title: Optional[str] = Field(description="Judul paper")
    authors: Optional[List[str]] = Field(description="Daftar nama penulis")
    year: Optional[str] = Field(description="Tahun publikasi")
    journal: Optional[str] = Field(description="Nama jurnal atau konferensi")
    doi: Optional[str] = Field(description="DOI")
    abstract: Optional[str] = Field(description="Abstrak dalam 2-3 kalimat")
    keywords: Optional[List[str]] = Field(description="Daftar keyword")
    institution: Optional[str] = Field(description="Institusi atau afiliasi penulis")


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    with st.expander("🔑 API Keys & Database", expanded=True):
        gemini_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=ENV_GEMINI_KEY or "",
            help="Wajib untuk Chatbot. Pre-filled dari .env jika tersedia.",
        )
        if ENV_GEMINI_KEY:
            st.caption("✅ Gemini key loaded from .env")

        exa_key_input = st.text_input(
            "Exa API Key",
            type="password",
            value=ENV_EXA_KEY or "",
            help="Opsional untuk fitur Web Search. Pre-filled dari .env jika tersedia.",
        )
        if ENV_EXA_KEY:
            st.caption("✅ Exa key loaded from .env")

        vector_db_choice = st.selectbox("Vector DB", ["ChromaDB", "FAISS"])
    
    st.session_state.gemini_api_key = gemini_key_input
    st.session_state.exa_api_key = exa_key_input
    
    st.markdown("### 📚 Data Dokumen")
    uploaded_files = st.file_uploader("Upload PDF papers", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    build_button = st.button("Build Knowledge Base", type="primary", use_container_width=True)
    
    st.markdown("### ⚡ Quick Actions")
    if "files" in st.session_state and st.session_state.files:
        selected_paper = st.selectbox("Pilih paper:", st.session_state.files, label_visibility="collapsed")
        
        st.caption("**Ekstraksi Dokumen**")
        colA, colB = st.columns(2)
        with colA:
            btn_summary = st.button("📝 Rangkum", use_container_width=True)
            btn_glossary = st.button("📖 Istilah", use_container_width=True)
            btn_insight = st.button("💡 Insight", use_container_width=True)
        with colB:
            btn_citation = st.button("🔗 Sitasi", use_container_width=True)
            btn_critique = st.button("⚖️ Kritik", use_container_width=True)
            btn_gap = st.button("🎯 Gap", use_container_width=True)
        
        st.caption("**Analisis Mendalam**")
        col1, col2 = st.columns(2)
        with col1:
            btn_methodology = st.button("🧪 Metodologi", use_container_width=True)
        with col2:
            btn_metadata = st.button("📊 Metadata", use_container_width=True)

        actions = {
            btn_summary: ("Rangkuman", SUMMARY_PROMPT),
            btn_glossary: ("Kamus Istilah", GLOSSARY_PROMPT),
            btn_citation: ("Sitasi", CITATION_PROMPT),
            btn_critique: ("Kritik Akademis", CRITIQUE_PROMPT),
            btn_insight: ("Research Insight", INSIGHT_PROMPT),
            btn_gap: ("Research Gap", GAP_PROMPT),
            btn_methodology: ("Analisis Metodologi", METHODOLOGY_PROMPT),
            btn_metadata: ("Metadata", METADATA_PROMPT),
        }
        
        for btn, action_data in actions.items():
            if btn:
                st.session_state.action_request = (selected_paper, action_data[0], action_data[1])
                break

        st.caption("**Multi-Paper**")
        btn_litreview = st.button("📑 Literature Review", use_container_width=True, help="Sintetis semua paper yang diupload")
        if btn_litreview:
            st.session_state.litreview_request = True
    else:
        st.info("Upload PDF untuk Quick Actions")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.session_title = "Chat Baru"

    st.markdown("### 📝 Riwayat Obrolan")
    
    if st.button("➕ Chat Baru", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.session_title = "Chat Baru"
        
    sessions = list_sessions()
    for s_id, s_title in sessions:
        col1, col2, col3 = st.columns([7, 1.5, 1.5])
        with col1:
            if st.button(f"💬 {s_title}", key=f"load_{s_id}", use_container_width=True):
                st.session_state.session_id = s_id
                st.session_state.session_title, st.session_state.messages = load_session(s_id)
        with col2:
            if st.button("📦", key=f"arc_{s_id}", help="Arsipkan chat ini"):
                archive_session(s_id)
                if st.session_state.session_id == s_id:
                    st.session_state.session_id = str(uuid.uuid4())[:8]
                    st.session_state.messages = []
                    st.session_state.session_title = "Chat Baru"
                st.rerun()
        with col3:
            if st.button("🗑️", key=f"del_{s_id}", help="Hapus chat ini"):
                delete_session(s_id)
                if st.session_state.session_id == s_id:
                    st.session_state.session_id = str(uuid.uuid4())[:8]
                    st.session_state.messages = []
                    st.session_state.session_title = "Chat Baru"
                st.rerun()

    # Archived sessions section
    archived = list_archived_sessions()
    if archived:
        with st.expander("📦 Obrolan Diarsipkan"):
            for s_id, s_title in archived:
                colA, colB, colC = st.columns([7, 1.5, 1.5])
                with colA:
                    if st.button(f"💬 {s_title}", key=f"aload_{s_id}", use_container_width=True):
                        st.session_state.session_id = s_id
                        st.session_state.session_title, st.session_state.messages = load_session(s_id)
                with colB:
                    if st.button("🔄", key=f"unarc_{s_id}", help="Kembalikan chat (Unarchive)"):
                        unarchive_session(s_id)
                        st.rerun()
                with colC:
                    if st.button("🗑️", key=f"adel_{s_id}", help="Hapus permanen"):
                        delete_session(s_id)
                        if st.session_state.session_id == s_id:
                            st.session_state.session_id = str(uuid.uuid4())[:8]
                            st.session_state.messages = []
                            st.session_state.session_title = "Chat Baru"
                        st.rerun()

# ============================================================
# VALIDATE API KEY & MODELS
# ============================================================
if not st.session_state.gemini_api_key:
    st.error("Google Gemini API key is missing. Silakan masukkan di sidebar.")
    st.stop()

st.title("🔬 AI Research Assistant")

@st.cache_resource
def load_models(api_key):
    return get_embeddings(api_key=api_key), get_llm(api_key=api_key)

try:
    embeddings, llm = load_models(st.session_state.gemini_api_key)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================
if build_button:
    if not uploaded_files:
        st.warning("Silakan upload minimal 1 PDF.")
    else:
        try:
            with st.spinner(f"Memproses dokumen dengan {vector_db_choice}..."):
                documents, chunks = load_and_split_papers(uploaded_files)
                if vector_db_choice == "FAISS":
                    vectorstore = create_faiss_vectorstore(chunks, embeddings)
                else:
                    vectorstore = create_chroma_vectorstore(chunks, embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.documents = documents
            st.session_state.files = [file.name for file in uploaded_files]
            st.session_state.messages = []
            
            with st.spinner("Merumuskan rekomendasi pertanyaan..."):
                st.session_state.dynamic_suggestions = generate_suggestions(documents, llm)
                
            st.success("Knowledge base berhasil dibuat!")
        except Exception as e:
            st.error(f"Gagal memproses dokumen! Pastikan API Key yang Anda masukkan valid dan aktif. Detail Error: {e}")

if "vectorstore" not in st.session_state:
    st.info("Upload research papers dan klik **Build Knowledge Base** di sidebar.")
    st.stop()

# ============================================================
# AGENT SETUP — rebuild only when vectorstore or keys change
# ============================================================
_vs_id = id(st.session_state.vectorstore)
_agent_stale = (
    "agent" not in st.session_state
    or st.session_state.get("_agent_vs_id") != _vs_id
    or st.session_state.get("_agent_api_key") != st.session_state.gemini_api_key
    or st.session_state.get("_agent_exa_key") != st.session_state.exa_api_key
)
if _agent_stale:
    st.session_state.agent = create_research_agent(
        llm,
        st.session_state.vectorstore,
        exa_api_key=st.session_state.exa_api_key,
    )
    st.session_state._agent_vs_id = _vs_id
    st.session_state._agent_api_key = st.session_state.gemini_api_key
    st.session_state._agent_exa_key = st.session_state.exa_api_key

agent = st.session_state.agent


if "files" in st.session_state and st.session_state.files:
    with st.expander("📄 Lihat Dokumen PDF (Preview)", expanded=False):
        if uploaded_files:
            selected_pdf = st.selectbox("Pilih dokumen untuk dibaca:", [f.name for f in uploaded_files], key="pdf_viewer")
            for f in uploaded_files:
                if f.name == selected_pdf:
                    pdf_bytes = f.getvalue()
                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.info("Silakan upload ulang file PDF di sidebar untuk membaca dokumen langsung di sini.")

    # ── Paper Metadata auto-extraction ──
    with st.expander("📊 Paper Metadata", expanded=False):
        if "paper_metadata" not in st.session_state:
            st.session_state.paper_metadata = {}
        
        for fname in st.session_state.files:
            if fname not in st.session_state.paper_metadata:
                # Extract from first page(s) only for speed
                docs_for_meta = [
                    d for d in st.session_state.documents
                    if d.metadata.get("filename") == fname
                ][:3]
                if docs_for_meta:
                    meta_text = "\n".join(d.page_content for d in docs_for_meta)[:3000]
                    try:
                        structured_llm = llm.with_structured_output(PaperMetadataModel)
                        meta_obj = structured_llm.invoke(
                            f"Ekstrak metadata dari dokumen akademis berikut:\n\n{meta_text}"
                        )
                        st.session_state.paper_metadata[fname] = meta_obj.dict() if meta_obj else None
                    except Exception:
                        st.session_state.paper_metadata[fname] = None

        for fname, meta in st.session_state.paper_metadata.items():
            st.markdown(f"**📄 {fname}**")
            if meta:
                cols = st.columns(2)
                cols[0].markdown(f"**Judul:** {meta.get('title') or 'N/A'}")
                cols[0].markdown(f"**Penulis:** {', '.join(meta.get('authors') or []) or 'N/A'}")
                cols[0].markdown(f"**Tahun:** {meta.get('year') or 'N/A'}")
                cols[0].markdown(f"**Jurnal:** {meta.get('journal') or 'N/A'}")
                cols[1].markdown(f"**DOI:** {meta.get('doi') or 'N/A'}")
                cols[1].markdown(f"**Institusi:** {meta.get('institution') or 'N/A'}")
                if meta.get('keywords'):
                    st.markdown(f"**Keywords:** {', '.join(meta['keywords'])}")
                if meta.get('abstract'):
                    st.markdown(f"**Abstrak:** {meta['abstract']}")
            else:
                st.caption("⚠️ Metadata tidak dapat diekstrak dari paper ini.")
            st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.markdown("💡 **Rekomendasi Pertanyaan:**")
cols = st.columns(3)

sug_labels = ["Apa kontribusi utamanya?", "Bandingkan metodenya", "Ringkas temuan akhirnya"]
if "dynamic_suggestions" in st.session_state:
    sug_labels = st.session_state.dynamic_suggestions

sug1 = cols[0].button(sug_labels[0], use_container_width=True)
sug2 = cols[1].button(sug_labels[1], use_container_width=True)
sug3 = cols[2].button(sug_labels[2], use_container_width=True)

question = st.chat_input("Tanya sesuatu atau suruh bandingkan paper...")

if sug1: 
    question = sug_labels[0]
if sug2: 
    question = sug_labels[1]
if sug3: 
    question = sug_labels[2]

action_req = st.session_state.pop("action_request", None)
litreview_req = st.session_state.pop("litreview_request", False)

if question or action_req or litreview_req:
    is_action = action_req is not None
    is_litreview = litreview_req and not is_action
    user_text = ""  # guard against UnboundLocalError
    if is_litreview:
        paper_list = ", ".join(st.session_state.files)
        user_text = f"Tolong buatkan **Literature Review** untuk semua paper: {paper_list}"
    elif is_action:
        paper_name, action_label, prompt_template = action_req
        user_text = f"Tolong buatkan **{action_label}** untuk paper: **{paper_name}**"
    elif question:
        user_text = question

    if not user_text:
        st.stop()

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_text)

    # Generate Title if it's the first message
    if len(st.session_state.messages) == 0:
        prompt = PromptTemplate.from_template(TITLE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        try:
            new_title = chain.invoke({"text": user_text})
            st.session_state.session_title = new_title.strip('"\' ')
        except Exception:
            st.session_state.session_title = "Percakapan"

    st.session_state.messages.append({
        "role": "user", 
        "content": user_text
    })

    # Display assistant message
    with st.chat_message("assistant"):
        try:
            if is_litreview:
                # Synthesize ALL uploaded documents
                all_docs = st.session_state.documents
                stream = quick_action_stream(all_docs, llm, LITERATURE_REVIEW_PROMPT)
                full_response = st.write_stream(stream)
            elif is_action:
                selected_documents = [
                    doc for doc in st.session_state.documents
                    if doc.metadata.get("filename") == paper_name
                ]
                stream = quick_action_stream(selected_documents, llm, prompt_template)
                full_response = st.write_stream(stream)
            else:
                # Agentic invocation
                # Truncate history to sliding window to prevent token overflow
                history = st.session_state.messages[:-1] # exclude current message
                history_window = history[-(MAX_HISTORY_TURNS * 2):] if MAX_HISTORY_TURNS > 0 else []
                langgraph_history = []
                for m in history_window:
                    if m["role"] == "user":
                        langgraph_history.append(HumanMessage(content=m["content"]))
                    else:
                        langgraph_history.append(AIMessage(content=m["content"]))
                        
                langgraph_history.append(HumanMessage(content=user_text))
                
                full_response = ""
                with st.spinner("Agent sedang berpikir..."):
                    placeholder = st.empty()
                    for chunk, _ in agent.stream({"messages": langgraph_history}, stream_mode="messages"):
                        if isinstance(chunk, AIMessageChunk) and chunk.content:
                            full_response += extract_text(chunk.content)
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })
            
            # Save session history
            save_session(st.session_state.session_id, st.session_state.messages, st.session_state.get("session_title", "Chat Baru"))

            # Refresh UI immediately so the newly generated title appears in the sidebar
            if len(st.session_state.messages) == 2:
                st.rerun()

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            print(traceback.format_exc())
