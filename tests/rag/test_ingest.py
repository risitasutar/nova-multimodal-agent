import pytest

from evaluation.fixtures.pdf_builder import build_pdf
from nova.config import get_settings
from nova.errors import DocumentError, DocumentTooLargeError, NoExtractableTextError, UnsupportedFileError
from nova.rag.ingest import clean_text, parse_pdf, sanitize_filename


def test_parse_report_keeps_one_based_pages_and_metadata(report_bytes):
    doc = parse_pdf(report_bytes, "northwind_annual_report_2025.pdf")
    assert doc.pages == 10 and len(doc.chunks) >= 10
    assert {c.page for c in doc.chunks} == set(range(1, 11))
    first = doc.chunks[0]
    assert first.page == 1 and first.document_id == doc.document_id
    assert first.filename == "northwind_annual_report_2025.pdf"
    assert len({c.chunk_id for c in doc.chunks}) == len(doc.chunks)
    net_income = [c for c in doc.chunks if "Net income was $28.1 million" in c.text]
    assert net_income and net_income[0].page == 3  # page numbers are real, not 0-based


def test_chunks_never_span_pages(report_bytes):
    doc = parse_pdf(report_bytes, "r.pdf")
    for c in doc.chunks:
        assert c.text.strip()
        assert len(c.text) <= get_settings().chunk_size


def test_injection_text_is_flagged(injection_bytes):
    doc = parse_pdf(injection_bytes, "contract.pdf")
    assert any("ignore previous instructions" in c.injection_flags for c in doc.chunks)


def test_rejects_non_pdf():
    with pytest.raises(UnsupportedFileError):
        parse_pdf(b"hello", "notes.txt")
    with pytest.raises(UnsupportedFileError):
        parse_pdf(b"<html>not a pdf</html>", "fake.pdf")


def test_rejects_malformed_pdf():
    with pytest.raises(DocumentError) as exc:
        parse_pdf(b"%PDF-1.4\n garbage garbage \n%%EOF", "broken.pdf")
    assert "damaged" in exc.value.user_message or "extractable" in exc.value.user_message


def test_rejects_pdf_without_text():
    empty_page = build_pdf([" "])
    with pytest.raises(NoExtractableTextError):
        parse_pdf(empty_page, "scan.pdf")


def test_size_and_page_limits(monkeypatch, report_bytes):
    monkeypatch.setenv("NOVA_MAX_PDF_PAGES", "3")
    get_settings.cache_clear()
    with pytest.raises(DocumentTooLargeError, match="pages"):
        parse_pdf(report_bytes, "r.pdf")
    monkeypatch.setenv("NOVA_MAX_PDF_MB", "0.001")
    get_settings.cache_clear()
    with pytest.raises(DocumentTooLargeError):
        parse_pdf(report_bytes, "r.pdf")


def test_filename_sanitized():
    assert sanitize_filename("../../etc/<script>.pdf") == "_script_.pdf"
    assert sanitize_filename(None) == "document.pdf"


def test_clean_text_dehyphenates_and_collapses():
    assert clean_text("inter-\nnational   trade\n\n\n\nok") == "international trade\n\nok"
