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
            return "No papers uploaded."
        docs = vectorstore.similarity_search(query, k=TOP_K)
        context = []
        for doc in docs:
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", 0) + 1
            context.append(f"[Source: {filename}, Page: {page}]\n{doc.page_content}")
        return "\n\n".join(context)

    @tool
    def compare_papers(paper_names: str, topic: str) -> str:
        """Compare multiple research papers on a given topic.

        Args:
            paper_names: Comma-separated list of exact filenames to compare.
                         Example: "paper_a.pdf, paper_b.pdf, paper_c.pdf"
            topic: The specific topic or aspect to compare across all papers.

        Supports comparing 2 or more papers at once.
        """
        if not vectorstore:
            return "No papers uploaded."
        try:
            # Parse comma-separated filenames
            names = [n.strip() for n in paper_names.split(",") if n.strip()]
            if len(names) < 2:
                return "Harap berikan minimal 2 nama file paper yang dipisahkan koma."

            # Retrieve a broad result set and post-filter by filename.
            # Compatible with both ChromaDB and FAISS (FAISS has no server-side filters).
            k_fetch = max(20, len(names) * 5)
            all_results = vectorstore.similarity_search(topic, k=k_fetch)

            sections = []
            missing = []
            for name in names:
                docs = [d for d in all_results if d.metadata.get("filename") == name][
                    :3
                ]
                if not docs:
                    missing.append(name)
                else:
                    content = "\n".join(d.page_content for d in docs)
                    sections.append(f"=== {name} ===\n{content}")

            if missing:
                return (
                    f"Tidak ditemukan konten dari: {', '.join(missing)}. "
                    "Pastikan nama file sudah benar dan sudah ada di knowledge base."
                )

            return (
                f"Comparing {len(names)} papers on topic: '{topic}'\n\n"
                + "\n\n".join(sections)
            )
        except Exception as e:
            return f"Error retrieving comparison: {e}"

    @tool
    def web_search(query: str) -> str:
        """Search the web for up-to-date information, news, or external state-of-the-art research that is not in the uploaded papers."""
        if not exa_client:
            return "Exa API key is not configured. Please provide it in the sidebar."
        try:
            results = exa_client.search_and_contents(
                query, type="auto", use_autoprompt=True, num_results=3
            )
            out = []
            for r in results.results:
                out.append(
                    f"Title: {r.title}\nURL: {r.url}\nContent Snippet: {r.text[:500]}"
                )
            return "\n\n".join(out)
        except Exception as e:
            return f"Web search failed: {e}"

    tools = [search_paper, compare_papers, web_search]
    agent = create_react_agent(llm, tools, prompt=AGENT_SYSTEM_PROMPT)
    return agent
