"""Reusable Pydantic validators — file content, path safety.

Consolidates validation logic that was previously scattered across route helpers
into composable Pydantic types and model validators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, model_validator

from presentation.api.constants import MAGIC_BYTES


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _reject_unsafe_path(v: str) -> str:
    """Reject paths containing '..' or absolute paths to prevent path traversal.

    This is a first-pass filter at the Pydantic schema level. Downstream
    services (``_validate_s3_key``, ``resolve_docs_dir``) perform their own
    checks as defense-in-depth.
    """
    if ".." in v:
        raise ValueError("Path must not contain '..'")
    if Path(v).is_absolute():
        raise ValueError("Absolute paths are not allowed")
    return v


# Annotated type — use as field type for automatic validation:
#   docs_dir: SafeRelativePath = "docs/"
SafeRelativePath = Annotated[str, AfterValidator(_reject_unsafe_path)]


# ---------------------------------------------------------------------------
# Structural file checks (pure functions)
# ---------------------------------------------------------------------------


def _is_zip_archive(data: bytes) -> bool:
    """Check for ZIP End Of Central Directory record."""
    if len(data) < 22:
        return False
    return b"PK\x05\x06" in data[-65557:]


def _is_valid_pdf(data: bytes) -> bool:
    """Check that PDF ends with %%EOF marker."""
    tail = data[-1024:] if len(data) > 1024 else data
    return b"%%EOF" in tail


def _is_ole2_document(data: bytes) -> bool:
    """Verify OLE2 compound document header structure."""
    if len(data) < 512:
        return False
    return data[26:28] == b"\x3e\x00" and data[30:32] == b"\xfe\xff" and data[32] == 0x09


# ---------------------------------------------------------------------------
# File content validation model
# ---------------------------------------------------------------------------


class FileContent(BaseModel):
    """Validate uploaded file extension and content.

    Use ``check_structural=True`` to additionally verify container format
    structure (PDF %%EOF, ZIP EOCD, OLE2 header) — raises 400 on mismatch.
    """

    data: bytes
    filename: str
    check_structural: bool = False

    @model_validator(mode="after")
    def _validate_content(self) -> FileContent:
        if not self.data:
            raise ValueError("File is empty")

        ext = Path(self.filename).suffix.lower()

        # 1. Extension must be known
        if ext not in MAGIC_BYTES:
            raise ValueError(f"Unsupported file extension: {ext}")

        # 2. Magic bytes must match declared extension
        expected = MAGIC_BYTES[ext]
        if expected and not any(self.data[: len(sig)] == sig for sig in expected):
            raise ValueError(f"File content does not match extension {ext}")

        # 3. Optional structural checks
        if self.check_structural:
            if ext == ".pdf" and not _is_valid_pdf(self.data):
                raise ValueError("File content is not a valid PDF (missing EOF marker)")
            if ext == ".docx" and not _is_zip_archive(self.data):
                raise ValueError("File content is not a valid ZIP-archive (.docx)")
            if ext == ".doc" and not _is_ole2_document(self.data):
                raise ValueError("File content is not a valid OLE2 compound document (.doc)")

        return self
