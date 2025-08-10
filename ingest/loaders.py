# ingest/loaders.py
from __future__ import annotations

import re
from typing import List, Iterable

from pydantic import BaseModel, Field, field_validator, model_validator

# Prefer PyMuPDF for cleaner extraction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from settings import settings

# -----------------------------------------------------------------------------
# Config for page labels (human-visible numbering)
# - PAGE_LABEL_OFFSET: human_label = pdf_page - PAGE_LABEL_OFFSET
# - PAGE_LABEL_BLANK_BEFORE: for physical pages <= this number, label is None
# -----------------------------------------------------------------------------
PAGE_LABEL_OFFSET: int = getattr(settings, "page_label_offset", 8)
PAGE_LABEL_BLANK_BEFORE: int = getattr(settings, "page_label_blank_before", 8)

# Fallbacks if not present in settings
DEFAULT_CHUNK_SIZE = getattr(settings, "chunk_size", 1200)
DEFAULT_CHUNK_OVERLAP = getattr(settings, "chunk_overlap", 180)

# -------- Heuristics ---------------------------------------------------------

# Numbered section headers like "3.16 2D Plotting" or "3.16.1 Types of Plotting"
SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+([^\n]+?)\s*$")

# Common header/footer lines to strip
HEADER_FOOTER_PATTERNS = [
    # "OpenModelica User’s Guide" / "OpenModelica User's Guide" / "OpenModelica Users Guide"
    re.compile(r"^\s*OpenModelica\s+User(?:['’`]?s)?\s+Guide.*$", re.IGNORECASE),
    # Standalone section titles from ToC-like lines: "3.14 ... 45"
    re.compile(r"^\s*\d+(?:\.\d+)*\s+.*\s+\d+\s*$"),
    # Footer with printed page number + chapter label:
    # "70 Chapter 3. OMEdit - OpenModelica Connection Editor"
    re.compile(r"^\s*\d+\s+Chapter\s+\d+\.\s+.*$", re.IGNORECASE),
    # Common front/back matter lines
    re.compile(r"^\s*CONTENTS\s*$", re.IGNORECASE),
    re.compile(r"^\s*Bibliography\s*\d*\s*$", re.IGNORECASE),
    re.compile(r"^\s*Index\s*\d*\s*$", re.IGNORECASE),
    # Roman numerals as page rows
    re.compile(r"^\s*[ivxlcdm]+\s*$", re.IGNORECASE),
    # Bare numeric page number line
    re.compile(r"^\s*\d+\s*$"),
]

SOFT_HYPHEN = "\u00ad"

# -------- Normalization utilities -------------------------------------------


def _normalize_page_text(raw: str) -> str:
    """
    Normalize common PDF artifacts:
      - remove soft hyphens
      - fix hyphenated line breaks: 'vari-\nable' -> 'variable'
      - standardize newlines
      - collapse 3+ newlines to 2 to preserve paragraphs
      - trim trailing spaces
    """
    txt = raw.replace("\r\n", "\n").replace("\r", "\n")
    txt = txt.replace(SOFT_HYPHEN, "")
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)  # de-hyphenate across line breaks
    txt = re.sub(r"\n{3,}", "\n\n", txt)  # squish huge blank blocks
    txt = re.sub(r"[ \t]+(\n)", r"\1", txt)  # trim line-end spaces
    return txt


def _iter_lines_preserve_blanks(text: str) -> Iterable[str]:
    """Yield lines, preserving blank lines (“”) so we can reconstruct paragraphs."""
    for raw in text.split("\n"):
        yield raw


def _strip_headers_footers(lines: Iterable[str]) -> Iterable[str]:
    """Remove obvious headers/footers but preserve blank lines (paragraphs)."""
    for raw in lines:
        line = raw.strip()
        if line == "":
            yield ""  # keep paragraph breaks
            continue
        if any(p.search(line) for p in HEADER_FOOTER_PATTERNS):
            continue
        yield raw


def _compress_blank_paragraphs(lines: Iterable[str]) -> List[str]:
    """Collapse multiple blanks to a single blank to avoid runaway empty lines."""
    out: List[str] = []
    blank = False
    for raw in lines:
        if raw.strip() == "":
            if not blank:
                out.append("")
                blank = True
        else:
            out.append(raw.strip())
            blank = False
    return out


# -------- Pydantic model -----------------------------------------------------


class Section(BaseModel):
    number: str  # e.g., "3.14.1"
    title: str  # e.g., "Creating a New Modelica Class"
    # Physical PDF page indices (1-based)
    start_page: int = Field(gt=0)
    end_page: int = Field(gt=0)
    # Human-visible page labels as integers (None for front matter)
    start_label: int | None
    end_label: int | None
    # Paragraph lines (with "" separating paragraphs)
    lines: List[str] = Field(default_factory=list)

    @field_validator("end_page")
    @classmethod
    def _end_ge_start(cls, v: int, info):
        start = info.data.get("start_page")
        if start is not None and v < start:
            raise ValueError("end_page must be >= start_page")
        return v

    @model_validator(mode="after")
    def _strip_title(self):
        self.title = self.title.strip()
        return self

    def bump_page(self, page: int, label: int | None) -> None:
        """Extend section page range & labels when we see more pages for it."""
        if page > self.end_page:
            self.end_page = page
            self.end_label = label

    def add_line(self, line: str) -> None:
        self.lines.append(line)

    def to_text(self) -> str:
        """Reconstruct section text with a clear header and preserved paragraphs."""
        header = f"{self.number} {self.title}".strip()
        paras: List[str] = []
        buf: List[str] = []
        for ln in self.lines:
            if ln.strip() == "":
                if buf:
                    paras.append(" ".join(buf).strip())
                    buf = []
            else:
                buf.append(ln.strip())
        if buf:
            paras.append(" ".join(buf).strip())
        body = "\n\n".join(paras)
        return f"{header}\n\n{body}".strip()


# -------- Page label mapping (offset-based) ----------------------------------


def _page_label_for(page_num_physical_1based: int) -> int | None:
    """
    Compute human page label from physical 1-based page index using a simple rule:
      - if page <= PAGE_LABEL_BLANK_BEFORE: return None
      - else: label = page - PAGE_LABEL_OFFSET
    """
    if page_num_physical_1based <= PAGE_LABEL_BLANK_BEFORE:
        return None
    return page_num_physical_1based - PAGE_LABEL_OFFSET


# -------- Page -> Section aggregation ----------------------------------------


def _pages_to_sections(pages: List[Document]) -> List[Section]:
    """
    Convert page Documents into section-sized buffers by scanning for SECTION_RE.
    Tracks both physical pages (1-based) and human labels (int or None).
    Ensures we never create a header-only section: content is attached.
    """
    sections: List[Section] = []
    current: Section | None = None

    for page_doc in pages:
        page0 = int(page_doc.metadata.get("page", 0))  # 0-based
        page_num = page0 + 1  # physical 1-based
        page_label = _page_label_for(page_num)

        normalized = _normalize_page_text(page_doc.page_content)
        lines = list(_iter_lines_preserve_blanks(normalized))
        lines = list(_strip_headers_footers(lines))
        lines = _compress_blank_paragraphs(lines)

        for line in lines:
            m = SECTION_RE.match(line.strip()) if line.strip() else None
            if m:
                # Flush previous section (if any)
                if current is not None:
                    current.bump_page(page_num, page_label)
                    sections.append(current)

                number, title = m.group(1), m.group(2)
                current = Section(
                    number=number,
                    title=title,
                    start_page=page_num,
                    end_page=page_num,
                    start_label=page_label,
                    end_label=page_label,
                )
            else:
                if current is None:
                    # Skip preface text outside numbered sections
                    continue
                current.add_line(line)

        if current is not None:
            current.bump_page(page_num, page_label)

    if current is not None:
        sections.append(current)

    # Post-process: merge any accidental header-only sections into their followers
    merged: List[Section] = []
    i = 0
    while i < len(sections):
        s = sections[i]
        body_has_text = any(ln.strip() for ln in s.lines)
        if not body_has_text and i + 1 < len(sections):
            nxt = sections[i + 1]
            # prepend header into next so it isn't lost; keep a paragraph break
            nxt.lines.insert(0, "")
            nxt.lines.insert(0, f"{s.number} {s.title}")
            nxt.start_page = min(s.start_page, nxt.start_page)
            nxt.start_label = (
                s.start_label if s.start_label is not None else nxt.start_label
            )
            merged.append(nxt)
            i += 2
        else:
            merged.append(s)
            i += 1

    return merged


# -------- Section -> Document splitting --------------------------------------


def _split_long_sections(sections: List[Section]) -> List[Document]:
    """
    Turn Sections into LangChain Documents.
    If a section is very long, apply a secondary character splitter to keep chunks small,
    but preserve section/page metadata (physical indices + human labels) on each sub-chunk.
    """
    docs: List[Document] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # prefer paragraph, then line, then space
    )

    def _base_meta(s: Section) -> dict:
        return {
            # Physical page indices (for #page= links)
            "pdf_page_start": s.start_page,
            "pdf_page_end": s.end_page,
            # Human-visible labels as integers
            "page_label_start": s.start_label,
            "page_label_end": s.end_label,
            # Section info
            "section_number": s.number,
            "section_title": s.title,
            # Source file path & doc id
            "source": settings.pdf_path,
            "doc_id": settings.pdf_path,
        }

    for s in sections:
        text = s.to_text().strip()
        parts = (
            [text]
            if len(text) <= DEFAULT_CHUNK_SIZE * 1.5
            else splitter.split_text(text)
        )
        total = len(parts)

        for idx, part in enumerate(parts):
            md = _base_meta(s) | {
                "section_part": idx,
                "section_parts": total,
                # "chunk_index" assigned in load_and_split_pdf for global order
            }
            docs.append(Document(page_content=part, metadata=md))

    return docs


# -------- Public API ---------------------------------------------------------


def load_and_split_pdf() -> List[Document]:
    """
    1) Load per-page text (PyMuPDFLoader keeps 0-based 'page' in metadata).
    2) Normalize & clean headers/footers while preserving paragraph structure.
    3) Aggregate by numbered section headings (e.g., 3.16 ...), tracking:
       - physical PDF pages (1-based) and
       - human page labels (int or None via simple offset rule).
    4) Split long sections with paragraph-aware overlap.
    5) Add stable metadata including doc_id and chunk_index.
    6) Return Documents with rich metadata for robust citations & neighbor lookup.
    """
    loader = PyMuPDFLoader(settings.pdf_path)  # switched from PyPDFLoader
    pages: List[Document] = loader.load()

    sections = _pages_to_sections(pages)
    docs = _split_long_sections(sections)

    # Assign a stable chunk_index to enable prev/next navigation later
    for i, d in enumerate(docs):
        d.metadata["chunk_index"] = i

    return docs
