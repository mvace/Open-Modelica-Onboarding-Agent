from typing import List, Dict
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import Tool
from settings import settings


def _get_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    return FAISS.load_local(
        settings.index_dir, embeddings, allow_dangerous_deserialization=True
    )


def _search_docs(query: str, k: int = 5) -> List[Dict]:
    vs = _get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.invoke(query)

    results: List[Dict] = []
    for d in docs:
        md = d.metadata or {}
        results.append(
            {
                "content": (d.page_content or "").strip()[:1800],
                "source": md.get("source"),
                "section_number": md.get("section_number"),
                "section_title": md.get("section_title"),
                "page_label_start": md.get("page_label_start"),
                "page_label_end": md.get("page_label_end"),
                "pdf_page_start": md.get("pdf_page_start"),
                "pdf_page_end": md.get("pdf_page_end"),
            }
        )
    return results


def doc_search_callable(query: str) -> str:
    """
    Tool entrypoint. Accepts a natural-language query and returns JSON string
    of passages.
    """
    hits = _search_docs(query, k=5)
    return json.dumps(hits, ensure_ascii=False)


# Expose as a LangChain Tool
DocSearchTool = Tool.from_function(
    name="doc_search",
    description=(
        "Search the OpenModelica User's Guide and return relevant passages.\n"
        "INPUT: a natural-language question.\n"
        "OUTPUT: JSON array of objects. Keys include:\n"
        "- content (str): The text content of the passage\n"
        "- page_label_start/page_label_end (str): Human labels shown by viewers like Adobe\n"
        "- pdf_page_start/pdf_page_end (int): Physical pages for #page= links\n"
        "- section_number/section_title (str): Section information\n"
        "- source (str): Source file path\n"
        'Example: [{"content": "...", "page_label_start": "61", "pdf_page_start": 53, '
        '"section_number": "3.14.1", "source": "..."}, ...]'
    ),
    func=doc_search_callable,
)
