import zipfile
from pathlib import Path
import pytest
from fastapi import HTTPException
from backend.app.api_sse import _validate_signature


def test_rejects_fake_pdf(tmp_path: Path):
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"not pdf")
    with pytest.raises(HTTPException):
        _validate_signature(path, ".pdf", b"not pdf")


def test_accepts_minimal_docx_signature(tmp_path: Path):
    path = tmp_path / "ok.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "x")
        archive.writestr("word/document.xml", "x")
    _validate_signature(path, ".docx", b"PK")
