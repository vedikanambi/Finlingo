from backend.app.core.config import Settings
from backend.app.services.regulatory_corpus import RegulatoryCorpusLoader


def test_ecfr_xml_extraction_preserves_section_context():
    loader = RegulatoryCorpusLoader(Settings(_env_file=None))
    text = loader._xml_text(
        "<ROOT><HEAD>§ 1026.1 Authority</HEAD><P>This regulation applies to consumer credit.</P></ROOT>"
    )
    assert "§ 1026.1 Authority" in text
    assert "[§ 1026.1 Authority] This regulation applies to consumer credit." in text
