from pathlib import Path
from backend.app.core.config import Settings
from backend.app.stages.stage1_document_parser import Stage1DocumentParser


def test_number_markers_are_not_standalone_clauses(tmp_path: Path):
    path = tmp_path / "contract.txt"
    path.write_text(
        "TERMS\n1. The borrower must pay €10 each month.\n2. The agreement renews automatically unless cancelled.\n",
        encoding="utf-8",
    )
    clauses = Stage1DocumentParser(Settings(_env_file=None)).parse(path)
    assert len(clauses) == 2
    assert clauses[0].text.startswith("1.")
    assert clauses[0].text != "1."
    assert clauses[0].section_heading == "TERMS"
    assert clauses[0].extraction_method == "txt"


def test_multilevel_number_without_terminal_dot_starts_clause(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "1.1 Definitions\nThe following terms apply to this agreement.\n"
        "1.2 Fees\nYou must pay a monthly administration fee.",
        encoding="utf-8",
    )
    clauses = Stage1DocumentParser(Settings(_env_file=None)).parse(path)
    assert len(clauses) == 2
    assert clauses[0].text.startswith("1.1 Definitions")
    assert clauses[1].text.startswith("1.2 Fees")


def test_scanned_pdf_uses_real_tesseract_ocr(tmp_path: Path):
    import shutil
    import pytest
    from PIL import Image, ImageDraw

    if shutil.which("tesseract") is None:
        pytest.skip("Tesseract is not installed")
    image = Image.new("RGB", (1800, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 100), "1. The borrower must pay ten euro each month.", fill="black", stroke_width=1)
    path = tmp_path / "scanned.pdf"
    image.save(path, "PDF", resolution=200.0)
    settings = Settings(_env_file=None, parser_ocr_dpi=200, parser_ocr_min_chars_per_page=40)
    clauses = Stage1DocumentParser(settings).parse(path)
    assert clauses
    assert any(clause.extraction_method == "ocr" for clause in clauses)
    assert any("borrower" in clause.text.lower() for clause in clauses)
