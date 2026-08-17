from exa_py import Exa
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from src.config import TOP_K
from src.prompts import AGENT_SYSTEM_PROMPT


def create_research_agent(llm, vectorstore, exa_api_key=None):
    """
    Creates a LangGraph ReAct agent equipped with specific tools.
    """

    # Instantiate Exa client once to avoid per-call overhead
    exa_client = Exa(api_key=exa_api_key) if exa_api_key else None

    @tool
    def search_paper(query: str) -> str:
        """Search the uploaded research papers for specific information to answer user questions."""
        if not vectorstore:
            return (
                "Tidak ada dokumen (PDF) yang diunggah ke knowledge base saat ini. "
                "Beri tahu pengguna untuk mengunggah PDF di sidebar jika ingin menganalisis paper, "
                "atau gunakan 'web_search' jika pengguna mencari informasi umum / online."
            )
        try:
            docs = vectorstore.similarity_search(query, k=TOP_K)
            if not docs:
                return "Tidak ditemukan konten yang cocok dengan query tersebut di dalam dokumen yang diunggah."

            context = []
            for doc in docs:
                filename = doc.metadata.get("filename", "Unknown")
                page = doc.metadata.get("page", 0) + 1
                context.append(
                    f"[Sumber: {filename}, Halaman {page}]\n{doc.page_content}"
                )
            return "\n\n".join(context)
        except Exception as e:
            return f"Error saat mencari dokumen: {e}"

    @tool
    def compare_papers(paper_names: str, topic: str) -> str:
        """Compare multiple research papers on a given topic.

        Args:
            paper_names: Comma-separated list of filenames to compare.
                         Example: "paper_a.pdf, paper_b.pdf, paper_c.pdf"
            topic: The specific topic or aspect to compare across all papers.

        Supports comparing 2 or more papers at once.
        """
        if not vectorstore:
            return (
                "Tidak ada dokumen yang diunggah ke knowledge base untuk dibandingkan."
            )
        try:
            # Parse comma-separated filenames and clean them
            raw_names = [n.strip() for n in paper_names.split(",") if n.strip()]
            if len(raw_names) < 2:
                return "Harap berikan minimal 2 nama file paper yang dipisahkan koma."

            # Fetch a sufficiently wide pool of results
            k_fetch = max(30, len(raw_names) * 8)
            all_results = vectorstore.similarity_search(topic, k=k_fetch)

            sections = []
            missing = []

            for raw_name in raw_names:
                clean_target = raw_name.lower().removesuffix(".pdf").strip()

                def _matches(doc_fname, _target=clean_target):
                    if not doc_fname:
                        return False
                    df = str(doc_fname).lower().removesuffix(".pdf").strip()
                    return _target == df or _target in df or df in _target

                matched_docs = [
                    d for d in all_results if _matches(d.metadata.get("filename", ""))
                ][:3]

                if not matched_docs:
                    missing.append(raw_name)
                else:
                    actual_name = matched_docs[0].metadata.get("filename", raw_name)
                    doc_texts = []
                    for d in matched_docs:
                        page_num = d.metadata.get("page", 0) + 1
                        doc_texts.append(f"(Hal {page_num}): {d.page_content}")
                    content = "\n".join(doc_texts)
                    sections.append(f"=== 📄 {actual_name} ===\n{content}")

            if missing and not sections:
                return (
                    f"Tidak ditemukan konten dari: {', '.join(missing)}. "
                    "Pastikan nama file sesuai dengan paper yang telah diunggah."
                )

            report = (
                f"Perbandingan {len(sections)} paper terkait topik: '{topic}'\n\n"
                + "\n\n".join(sections)
            )
            if missing:
                report += f"\n\n[Catatan: Paper berikut tidak ditemukan di indeks: {', '.join(missing)}]"
            return report
        except Exception as e:
            return f"Error saat membandingkan paper: {e}"

    @tool
    def web_search(query: str) -> str:
        """Search the web for up-to-date information, news, or external state-of-the-art research that is not in the uploaded papers."""
        if not exa_client:
            return "Exa API key belum dikonfigurasi. Silakan masukkan Exa API Key di sidebar untuk mengaktifkan web search."
        try:
            results = exa_client.search_and_contents(
                query, type="auto", use_autoprompt=True, num_results=3
            )
            if not results or not getattr(results, "results", None):
                return (
                    "Tidak ditemukan hasil web yang relevan untuk pencarian tersebut."
                )

            out = []
            for r in results.results:
                title = getattr(r, "title", "No Title") or "No Title"
                url = getattr(r, "url", "") or ""
                text = getattr(r, "text", "") or ""
                snippet = text[:500].strip() if text else "Tidak ada ringkasan teks."
                out.append(f"Title: {title}\nURL: {url}\nContent Snippet: {snippet}")
            return "\n\n".join(out)
        except Exception as e:
            return f"Web search gagal: {e}"

    tools = [search_paper, compare_papers, web_search]
    agent = create_react_agent(llm, tools, prompt=AGENT_SYSTEM_PROMPT)
    return agent
