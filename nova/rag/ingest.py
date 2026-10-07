"""
PDF ingestion: validate → parse → clean → chunk (per page) → attach metadata.

Chunks never span pages, so every chunk has exactly one, 1-based page number and
citations can never point at the wrong page.
"""

from __future__ import annotations

import hashlib
import io
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from nova.config import get_settings
from nova.errors import DocumentError, DocumentTooLargeError, NoExtractableTextError, UnsupportedFileError
from nova.safety import find_injection_markers

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ \-()]+")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int  # 1-based
    chunk_index: int
    text: str
    injection_flags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    file_sha256: str
    pages: int
    chunks: list[Chunk]


def sanitize_filename(name: str | None) -> str:
    base = (name or "document.pdf").replace("\\", "/").split("/")[-1]
    base = _SAFE_NAME.sub("_", base).strip() or "document.pdf"
    return base[:120]


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Drop header/footer lines that repeat on most pages (page furniture, not content)."""
    if len(pages) < 3:
        return pages
    counts = Counter(
        line for page in pages for line in {ln.strip() for ln in page.splitlines()} if line.strip()
    )
    threshold = max(2, int(len(pages) * 0.6))
    repeated = {line for line, n in counts.items() if n >= threshold and len(line) < 80}
    if not repeated:
        return pages
    return ["\n".join(ln for ln in p.splitlines() if ln.strip() not in repeated) for p in pages]


def validate_pdf_bytes(data: bytes, filename: str) -> None:
    settings = get_settings()
    if not filename.lower().endswith(".pdf"):
        raise UnsupportedFileError(f"rejected extension: {filename}")
    if not data:
        raise DocumentError("empty upload", user_message="The uploaded file is empty.")
    max_bytes = int(settings.max_pdf_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise DocumentTooLargeError(
            f"{len(data)} bytes > {max_bytes}",
            user_message=f"This PDF is larger than the {settings.max_pdf_mb:g} MB limit.",
        )
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise UnsupportedFileError("missing %PDF- header", user_message="This file is not a valid PDF.")


def extract_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    settings = get_settings()
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError("encrypted", user_message="This PDF is password-protected.")
        n_pages = len(reader.pages)
        if n_pages > settings.max_pdf_pages:
            raise DocumentTooLargeError(
                f"{n_pages} pages",
                user_message=f"This PDF has {n_pages} pages; the limit is {settings.max_pdf_pages}.",
            )
        return [page.extract_text() or "" for page in reader.pages]
    except DocumentError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise DocumentError(repr(exc), user_message="This PDF is damaged or could not be read.") from exc


def parse_pdf(data: bytes, filename: str | None) -> ParsedDocument:
    settings = get_settings()
    safe_name = sanitize_filename(filename)
    validate_pdf_bytes(data, safe_name)
    pages = _strip_repeated_lines([clean_text(p) for p in extract_pages(data)])

    document_id = uuid.uuid4().hex
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for piece in splitter.split_text(page_text):
            if len(piece.strip()) < 20:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id[:8]}-p{page_no}-c{len(chunks)}",
                    document_id=document_id,
                    filename=safe_name,
                    page=page_no,
                    chunk_index=len(chunks),
                    text=piece.strip(),
                    injection_flags=find_injection_markers(piece),
                )
            )
            if len(chunks) > settings.max_chunks_per_document:
                raise DocumentTooLargeError(
                    "too many chunks",
                    user_message="This PDF contains too much text to index. Try a shorter document.",
                )
    if not chunks:
        raise NoExtractableTextError(f"no text in {safe_name}")
    return ParsedDocument(
        document_id=document_id,
        filename=safe_name,
        file_sha256=hashlib.sha256(data).hexdigest(),
        pages=len(pages),
        chunks=chunks,
    )
