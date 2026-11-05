"""Tests for FileContent and SafeRelativePath validators — FINDING-007 (HIGH).

Security-critical validators: path traversal prevention, file content validation.
"""

from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from pydantic import ValidationError
import pytest

from presentation.api.schemas.admin_act_versions import ActVersionUpdateRequest
from presentation.api.schemas.validators import FileContent, _reject_unsafe_path


# ---------------------------------------------------------------------------
# SafeRelativePath
# ---------------------------------------------------------------------------


def test_act_version_update_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ActVersionUpdateRequest.model_validate({"effective_from": "2026-01-01", "unexpected": True})

    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
    assert exc_info.value.errors()[0]["loc"] == ("unexpected",)


class TestSafeRelativePathRejectsDotdot:
    """Path traversal via '..' must be rejected."""

    def test_simple_dotdot(self) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            _reject_unsafe_path("../etc/passwd")

    def test_nested_dotdot(self) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            _reject_unsafe_path("docs/../../../etc/shadow")

    def test_dotdot_in_middle(self) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            _reject_unsafe_path("docs/../secrets/file.txt")

    def test_trailing_dotdot(self) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            _reject_unsafe_path("docs/..")

    def test_double_dotdot(self) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            _reject_unsafe_path("a/../../b")


class TestSafeRelativePathRejectsAbsolute:
    """Absolute paths must be rejected."""

    def test_unix_absolute(self) -> None:
        with pytest.raises(ValueError, match="Absolute"):
            _reject_unsafe_path("/etc/passwd")

    def test_root_relative(self) -> None:
        with pytest.raises(ValueError, match="Absolute"):
            _reject_unsafe_path("/home/user/file.txt")


class TestSafeRelativePathAccepts:
    """Valid relative paths are accepted."""

    def test_simple_name(self) -> None:
        assert _reject_unsafe_path("file.txt") == "file.txt"

    def test_nested_relative(self) -> None:
        assert _reject_unsafe_path("docs/report.pdf") == "docs/report.pdf"

    def test_deeply_nested(self) -> None:
        assert _reject_unsafe_path("a/b/c/d/file.md") == "a/b/c/d/file.md"

    def test_single_dot_ok(self) -> None:
        assert _reject_unsafe_path("./file.txt") == "./file.txt"

    def test_empty_string(self) -> None:
        assert _reject_unsafe_path("") == ""


# ---------------------------------------------------------------------------
# FileContent — magic bytes validation
# ---------------------------------------------------------------------------


class TestFileContentMagicBytes:
    """Magic bytes must match declared extension."""

    def test_pdf_valid(self) -> None:
        fc = FileContent(data=b"%PDF-1.4 test", filename="doc.pdf")
        assert fc.filename == "doc.pdf"

    def test_pdf_wrong_magic(self) -> None:
        with pytest.raises(ValidationError, match="does not match"):
            FileContent(data=b"PK\x03\x04not-a-pdf", filename="doc.pdf")

    def test_docx_valid(self) -> None:
        fc = FileContent(data=b"PK\x03\x04contents", filename="doc.docx")
        assert fc.filename == "doc.docx"

    def test_docx_wrong_magic(self) -> None:
        with pytest.raises(ValidationError, match="does not match"):
            FileContent(data=b"%PDF-1.4 fake", filename="doc.docx")

    def test_rtf_valid(self) -> None:
        fc = FileContent(data=b"{\\rtf1 ansi", filename="doc.rtf")
        assert fc.filename == "doc.rtf"

    def test_rtf_wrong_magic(self) -> None:
        with pytest.raises(ValidationError, match="does not match"):
            FileContent(data=b"random bytes", filename="doc.rtf")

    @pytest.mark.parametrize("filename", ["legacy.doc", "legacy.DOC"])
    @pytest.mark.parametrize("check_structural", [False, True])
    def test_legacy_doc_is_rejected(self, filename: str, check_structural: bool) -> None:
        with pytest.raises(ValidationError, match="Unsupported file extension"):
            FileContent(data=b"\xd0\xcf\x11\xe0 ole2", filename=filename, check_structural=check_structural)

    def test_txt_no_magic_check(self) -> None:
        fc = FileContent(data=b"any content", filename="doc.txt")
        assert fc.filename == "doc.txt"

    def test_md_no_magic_check(self) -> None:
        fc = FileContent(data=b"# Heading", filename="doc.md")
        assert fc.filename == "doc.md"

    def test_unknown_extension(self) -> None:
        with pytest.raises(ValidationError, match="Unsupported file extension"):
            FileContent(data=b"data", filename="file.xyz")


class TestFileContentStructural:
    """Structural validation when check_structural=True."""

    def test_pdf_missing_eof(self) -> None:
        with pytest.raises(ValidationError, match="missing EOF marker"):
            FileContent(data=b"%PDF-1.4 noeof", filename="doc.pdf", check_structural=True)

    def test_pdf_valid_with_structural(self) -> None:
        fc = FileContent(data=b"%PDF-1.4 content %%EOF", filename="doc.pdf", check_structural=True)
        assert fc.filename == "doc.pdf"

    def test_docx_not_zip(self) -> None:
        with pytest.raises(ValidationError, match="not a valid ZIP-archive"):
            FileContent(data=b"PK\x03\x04not-zip", filename="doc.docx", check_structural=True)

    def test_structural_disabled_passes(self) -> None:
        fc = FileContent(data=b"%PDF-1.4 noeof", filename="doc.pdf", check_structural=False)
        assert fc.filename == "doc.pdf"


class TestFileContentEmpty:
    """Empty file handling."""

    def test_empty_pdf(self) -> None:
        with pytest.raises(ValidationError, match="File is empty"):
            FileContent(data=b"", filename="doc.pdf")

    def test_empty_txt(self) -> None:
        with pytest.raises(ValidationError, match="File is empty"):
            FileContent(data=b"", filename="doc.txt")

    def test_empty_docx(self) -> None:
        with pytest.raises(ValidationError, match="File is empty"):
            FileContent(data=b"", filename="doc.docx")
