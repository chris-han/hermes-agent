import json

from tools.documents_tool import _parse_pages, read_document


def test_parse_pages_handles_ranges_and_single_pages():
    assert _parse_pages("1-3,5,7-8", 10) == [0, 1, 2, 4, 6, 7]


def test_read_document_rejects_missing_file(tmp_path):
    result = json.loads(read_document(str(tmp_path / "missing.pdf")))
    assert result["status"] == "error"
    assert "File not found" in result["error"]
