# 🔬 AI Research Assistant

AI Research Assistant adalah aplikasi berbasis [Streamlit](https://streamlit.io/) yang dirancang untuk membantu para peneliti, mahasiswa, dan akademisi dalam membedah, memahami, dan mensintesis berbagai jurnal/dokumen riset akademis secara interaktif.

Aplikasi ini menggunakan sistem **RAG (Retrieval-Augmented Generation)** yang dikombinasikan dengan pendekatan **Agentic (ReAct Agent)**, sehingga AI mampu menavigasi dokumen PDF, membandingkan antar *paper*, sekaligus melakukan pencarian web secara otomatis.

---

## ✨ Fitur Lengkap

### 💬 Research Chat
Chat interaktif berbasis LangGraph ReAct Agent. AI akan mencari bagian relevan dari PDF dan melakukan web search jika diperlukan. Setiap jawaban disertai **sitasi sumber** (nama file + halaman).

### 📚 Paper Knowledge Base
Upload satu atau beberapa PDF dan bangun knowledge base secara lokal menggunakan **ChromaDB** atau **FAISS**. Dokumen diproses dengan NLTK text splitter untuk chunking yang cerdas.

### 🔎 Semantic Search
Retrieval berbasis embedding (Google Gemini Embedding) untuk menemukan bagian dokumen yang paling relevan dengan pertanyaan pengguna.

### 🧠 Conversation Memory
Riwayat percakapan tersimpan otomatis dalam format JSON lokal dengan judul sesi yang di-*generate* otomatis oleh AI. Sliding window history (10 turn terakhir) untuk efisiensi token.

### 📄 PDF Upload & Viewer
Upload PDF langsung dari sidebar dan baca dokumen aslinya secara *side-by-side* dengan chat melalui **PDF Viewer** terintegrasi.

### ⚡ Quick Actions (Single-Paper)
Tombol satu klik untuk analisis mendalam sebuah paper:

| Tombol | Fungsi |
|--------|--------|
| **Rangkum** | Ringkasan terstruktur (Problem, Metodologi, Dataset, Hasil, Kontribusi) |
| **Sitasi** | Daftar pustaka dalam format APA, IEEE, dan MLA |
| **Istilah** | Kamus mini istilah teknis dan jargon ilmiah |
| **Kritik** | Kritik akademis: kelemahan, bias, limitasi eksperimen |
| **💡 Insight** | 5 research insight non-obvious yang paling impactful |
| **🎯 Gap** | Identifikasi celah penelitian & rekomendasi riset selanjutnya |
| **🧪 Metodologi** | Analisis mendalam: desain, data, algoritma, metrik, reproducibility |
| **📊 Metadata** | Ekstraksi metadata terstruktur (Judul, Penulis, Tahun, DOI, dsb) |

### 📑 Literature Review (Multi-Paper)
Sintetis **semua paper** yang diupload menjadi satu Literature Review kohesif: tren, perbandingan pendekatan, konsensus, kontradiksi, dan research gap.

### ⚖️ Paper Comparison
Bandingkan **2 atau lebih paper** sekaligus pada topik tertentu. Cukup sebutkan nama-nama file yang ingin dibandingkan dalam chat.

### 📊 Paper Metadata Auto-Extraction
Metadata bibliografi (Judul, Penulis, Tahun, Jurnal, DOI, Abstrak, Keywords, Institusi) diekstrak secara otomatis setelah knowledge base selesai dibangun dan ditampilkan dalam panel terpisah.

---

## 🚀 Cara Menjalankan

### 1. Persiapan Lingkungan

Pastikan menggunakan **Python 3.11+**.

```bash
# Masuk ke direktori proyek
cd ai-research-assistant

# Buat virtual environment
python -m venv .venv

# Aktifkan virtual environment
# Mac/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt
# Atau jika menggunakan uv:
uv sync
```

### 2. Konfigurasi API Keys

Aplikasi membutuhkan **Google Gemini API Key** (gratis via [Google AI Studio](https://aistudio.google.com/)). Opsional: **Exa API Key** untuk fitur pencarian web ([exa.ai](https://exa.ai/)).

**Opsi A — via `.env` file (direkomendasikan untuk pengembang):**

Salin `.env.example` menjadi `.env` lalu isi dengan key Anda:
```bash
cp .env.example .env
```
```env
GOOGLE_API_KEY=AIzaSy...
EXA_API_KEY=...         # opsional
TEMPERATURE=0.2         # opsional
CHUNK_SIZE=2000         # opsional
CHUNK_OVERLAP=200       # opsional
```
Jika `.env` sudah terisi, field API Key di sidebar akan **otomatis ter-isi** saat aplikasi dibuka.

**Opsi B — via Sidebar UI (untuk pengguna eksternal):**

Masukkan API Key langsung di panel kiri (*sidebar*) setiap kali membuka aplikasi.

### 3. Jalankan Aplikasi

```bash
streamlit run main.py
```

Aplikasi akan terbuka otomatis di browser, biasanya di `http://localhost:8501`.

---

## 📖 Panduan Penggunaan

1. **Masukkan API Key** di sidebar (atau otomatis dari `.env`)
2. **Upload PDF** di bagian "Data Dokumen" di sidebar
3. **Klik "Build Knowledge Base"** — tunggu hingga proses selesai
4. **Baca metadata** paper Anda di panel `📊 Paper Metadata` yang muncul otomatis
5. **Mulai chat** atau gunakan tombol Quick Actions di sidebar
6. Untuk **membandingkan beberapa paper**, cukup tulis di chat: *"Bandingkan metodologi paper_a.pdf, paper_b.pdf, dan paper_c.pdf"*
7. Untuk **Literature Review** dari semua paper, klik tombol `📑 Literature Review` di sidebar
8. Untuk **memulai sesi baru**, klik `➕ Chat Baru` di bagian Riwayat Obrolan

---

## 🛠️ Stack Teknologi

| Komponen | Teknologi |
|----------|-----------|
| **Front-end** | Streamlit |
| **LLM** | Google Gemini (`gemini-3.5-flash` / `gemini-3.5-flash-lite`) via LangChain |
| **Embeddings** | Google Gemini Embedding (`gemini-embedding-2`) |
| **Vector Database** | ChromaDB & FAISS (keduanya didukung penuh) |
| **Agent Orchestration** | LangGraph `create_react_agent` |
| **PDF Parser** | PyMuPDF |
| **Text Splitter** | NLTK Text Splitter |
| **Web Search** | Exa.ai |
| **History Storage** | JSON lokal |

---

## 📁 Struktur Proyek

```
ai-research-assistant/
├── main.py              # Entry point — Streamlit UI & orchestration
├── src/
│   ├── agent.py         # LangGraph ReAct agent + tools
│   ├── config.py        # Konfigurasi dari .env
│   ├── documents.py     # PDF loader & NLTK splitter
│   ├── helpers.py       # Utility functions (extract_text, build_chat_history)
│   ├── history.py       # JSON session persistence
│   ├── models.py        # LLM & Embeddings factory
│   ├── prompts.py       # Semua system/task prompts
│   └── rag.py           # Vectorstore factory & quick action stream
├── chat_history/        # Riwayat obrolan tersimpan (auto-created)
├── .env                 # API keys & config (jangan di-commit!)
├── .env.example         # Template konfigurasi
├── pyproject.toml       # Dependensi proyek
└── README.md            # Dokumentasi projek
```
