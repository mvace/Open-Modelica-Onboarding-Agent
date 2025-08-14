"""Chain for generating answers from OpenModelica documentation with accurate page citations."""

import json
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from settings import settings


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an onboarding assistant for OpenModelica. "
            "Answer using ONLY the provided passages. "
            "If ANY relevant passages are provided, you MUST summarize what they say about the question. "
            "If parts are missing, briefly state what is missing. "
            "Only say you don't know when ZERO passages are provided. "
            "Do NOT include a 'References' line yourself; it will be appended by the system.",
        ),
        (
            "human",
            "User question:\n{question}\n\n"
            "Retrieved passages (JSON list):\n{passages_json}\n\n"
            "Write a concise answer for the user. Use bullet points when helpful. "
            "If the question is broad (e.g., starts with 'tell me about', 'what is', or 'overview'), "
            "produce a short overview (3–6 bullets) covering key subtopics present in the passages. "
            "Do not invent facts not present in the passages.",
        ),
    ]
)


def _pdf_link(text: str, idx: int) -> str:
    url = getattr(settings, "pdf_url", None)
    return f"[{text}]({url}#page={idx})" if url else text


def _fmt_ref(lbl_s: int | None, lbl_e: int | None, idx_s: int, idx_e: int) -> str:
    txt_s = str(lbl_s) if lbl_s is not None else str(idx_s)
    txt_e = str(lbl_e) if lbl_e is not None else str(idx_e)
    if idx_s == idx_e and txt_s == txt_e:
        return f"page {_pdf_link(txt_s, idx_s)}"
    return f"pages {_pdf_link(f'{txt_s}–{txt_e}', idx_s)}"


def _extract_range(p: Dict) -> Tuple[int | None, int | None, int, int]:
    ls = p.get("page_label_start")
    le = p.get("page_label_end")
    is_ = p.get("pdf_page_start")
    ie = p.get("pdf_page_end")
    if ls is None:
        ls = is_
    if le is None:
        le = ie
    if is_ is None:
        is_ = ls or 0
    if ie is None:
        ie = le or is_
    return ls, le, int(is_), int(ie)


def _merge_overlaps(
    ranges: List[Tuple[int | None, int | None, int, int]],
) -> List[Tuple[int | None, int | None, int, int]]:
    if not ranges:
        return []
    merged: List[Tuple[int | None, int | None, int, int]] = []
    for r in sorted(ranges, key=lambda x: (x[2], x[3])):
        if not merged:
            merged.append(r)
            continue
        ls, le, ps, pe = merged[-1]
        cur_ls, cur_le, cur_ps, cur_pe = r
        if cur_ps <= pe:
            new_ls = (
                ls if (ls is not None and (cur_ls is None or ls <= cur_ls)) else cur_ls
            )
            new_le = (
                le if (le is not None and (cur_le is None or le >= cur_le)) else cur_le
            )
            merged[-1] = (new_ls, new_le, min(ps, cur_ps), max(pe, cur_pe))
        else:
            merged.append(r)
    return merged


def build_references(passages: List[Dict]) -> str:
    ranges: List[Tuple[int | None, int | None, int, int]] = []
    for p in passages:
        ls, le, is_, ie = _extract_range(p)
        if not is_:
            continue
        ranges.append((ls, le, is_, ie))
    if not ranges:
        return ""
    merged = _merge_overlaps(ranges)

    def sort_key(t):
        ls, _, is_, _ = t
        return (0, ls) if ls is not None else (1, is_)

    ordered = sorted(merged, key=sort_key)
    parts = [_fmt_ref(*t) for t in ordered]
    return "References: " + ", ".join(parts) if parts else ""


def answer_with_citations(question: str, passages: List[Dict]) -> str:
    """Generate an answer grounded in passages, then append formatted citations."""
    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
    passages_json = json.dumps(passages, ensure_ascii=False)
    msg = PROMPT.format_messages(question=question, passages_json=passages_json)
    resp = llm.invoke(msg)
    answer = (resp.content or "").strip()
    refs = build_references(passages)
    return f"{answer}\n\n{refs}" if refs else answer
