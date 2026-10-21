"""Reusable Pydantic validators — file content, path safety.

Consolidates validation logic that was previously scattered across route helpers
into composable Pydantic types and model validators.
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ValidationError, model_validator
from fastapi.exceptions import RequestValidationError

from presentation.api.constants import MAGIC_BYTES
from infrastructure.storage.docx_validation import validate_docx_archive


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


def _is_valid_pdf(data: bytes) -> bool:
    """Check that PDF ends with %%EOF marker."""
    tail = data[-1024:] if len(data) > 1024 else data
    return b"%%EOF" in tail


# ---------------------------------------------------------------------------
# File content validation model
# ---------------------------------------------------------------------------


class FileContent(BaseModel):
    """Validate uploaded file extension and content.

    Use ``check_structural=True`` to additionally verify container format
    structure (PDF %%EOF, ZIP EOCD) — raises 400 on mismatch.
    """

    data: bytes
    filename: str
    check_structural: bool = False

    @model_validator(mode="after")
    def validate_content(self) -> FileContent:
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
            if ext == ".docx":
                validate_docx_archive(BytesIO(self.data))

        return self


def validate_uploaded_file(data: bytes, filename: str) -> FileContent:
    """Translate user file validation failures into the public 422 contract."""
    try:
        return FileContent(data=data, filename=filename, check_structural=True)
    except ValidationError as exc:
        errors = [
            {"loc": ("body", "file"), "msg": e["msg"], "type": e["type"]}
            for e in exc.errors(include_input=False)
        ]
        raise RequestValidationError(errors) from exc
