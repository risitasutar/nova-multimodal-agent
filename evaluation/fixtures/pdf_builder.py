"""
Minimal dependency-free multi-page PDF writer (Helvetica text, word-wrapped).
Used to generate evaluation fixtures and test documents reproducibly.
"""

from __future__ import annotations

import textwrap

LINE_HEIGHT = 15
TOP = 750
BOTTOM = 60
WRAP = 92


def _escape(text: str) -> str:
    text = text.encode("latin-1", "replace").decode("latin-1")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_stream(lines: list[str]) -> str:
    ops = ["BT", "/F1 11 Tf", f"{LINE_HEIGHT} TL", f"60 {TOP} Td"]
    for line in lines:
        ops.append(f"({_escape(line)}) Tj T*")
    ops.append("ET")
    return "\n".join(ops)


def build_pdf(pages: list[str]) -> bytes:
    """Each item of `pages` is the text of one page (paragraphs separated by blank lines)."""
    max_lines = (TOP - BOTTOM) // LINE_HEIGHT
    page_lines: list[list[str]] = []
    for text in pages:
        lines: list[str] = []
        for para in text.strip().split("\n"):
            lines.extend(textwrap.wrap(para, WRAP) or [""])
        if len(lines) > max_lines:
            raise ValueError(f"page has {len(lines)} lines, max {max_lines}")
        page_lines.append(lines)

    n = len(page_lines)
    # Object numbering: 1 catalog, 2 pages, 3 font, then (page, content) pairs.
    objs: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    kids = []
    for i, lines in enumerate(page_lines):
        page_obj, content_obj = 4 + 2 * i, 5 + 2 * i
        kids.append(f"{page_obj} 0 R")
        stream = _page_stream(lines)
        objs[page_obj] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj} 0 R >>"
        )
        objs[content_obj] = f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream"
    objs[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {n} >>"

    out = "%PDF-1.4\n"
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out.encode("latin-1"))
        out += f"{num} 0 obj\n{objs[num]}\nendobj\n"
    xref = len(out.encode("latin-1"))
    total = max(objs) + 1
    out += f"xref\n0 {total}\n0000000000 65535 f \n"
    out += "".join(f"{offsets[i]:010d} 00000 n \n" for i in range(1, total))
    out += f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")
