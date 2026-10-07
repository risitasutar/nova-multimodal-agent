"""
Structured exceptions. Each carries a stable `code`, an HTTP status for the API and a
`user_message` that is safe to show to end users. Technical detail goes to the logs only.
"""

from __future__ import annotations


class NovaError(Exception):
    code = "internal_error"
    http_status = 500
    user_message = "Nova hit an unexpected problem. Please try again."

    def __init__(self, detail: str | None = None, *, user_message: str | None = None):
        super().__init__(detail or self.user_message)
        self.detail = detail or ""
        if user_message:
            self.user_message = user_message

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.user_message}


class InputValidationError(NovaError):
    code = "invalid_input"
    http_status = 422
    user_message = "That request could not be processed."


class LLMUnavailableError(NovaError):
    code = "llm_unavailable"
    http_status = 503
    user_message = "Nova cannot reach the language model server. Please start Ollama and try again."


class DocumentError(NovaError):
    code = "document_error"
    http_status = 422
    user_message = "This PDF could not be processed."


class UnsupportedFileError(DocumentError):
    code = "unsupported_file"
    http_status = 415
    user_message = "Only PDF files are supported."


class DocumentTooLargeError(DocumentError):
    code = "document_too_large"
    http_status = 413


class NoExtractableTextError(DocumentError):
    code = "no_extractable_text"
    user_message = "This PDF does not contain extractable text (it may be a scanned image)."


class IndexIntegrityError(DocumentError):
    code = "index_integrity"
    http_status = 500
    user_message = "A stored document index failed its integrity check and was not loaded."


class ThreadNotFoundError(NovaError):
    code = "thread_not_found"
    http_status = 404
    user_message = "That conversation does not exist."


class ApprovalStateError(NovaError):
    code = "no_pending_approval"
    http_status = 409
    user_message = "There is no pending approval for this conversation."


def is_connection_error(exc: BaseException) -> bool:
    """True when an exception chain looks like the LLM server is unreachable."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = type(cur).__name__.lower()
        text = str(cur).lower()
        if "connect" in name or "connection refused" in text or "failed to connect" in text:
            return True
        cur = cur.__cause__ or cur.__context__
    return False
