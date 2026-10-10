"""引用跳转必须定位实际上传原文的物理页，不能接受任意路径。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hashlib import sha256
import tempfile
import unittest
import pymupdf
from src.frontend.components.documents import read_pdf_page


class TestCitationPage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with pymupdf.open() as pdf:
            for text in ("First physical page", "Second physical page"):
                pdf.new_page().insert_text((40, 40), text)
            self.data = pdf.tobytes()
        identifier = sha256(self.data).hexdigest()
        self.source = self.root / identifier / "paper.pdf"
        self.source.parent.mkdir()
        self.source.write_bytes(self.data)
        self.reference = {"source_file": "paper.pdf", "metadata": {
            "doc_id": identifier, "page_number": 2}}

    def test_returns_actual_second_page_and_unchanged_pdf(self):
        result = read_pdf_page(self.root, self.reference)
        self.assertEqual(result["pdf"], self.data)
        self.assertEqual(result["page_number"], 2)
        self.assertTrue(result["image"].startswith(b"\x89PNG"))
        with pymupdf.open(stream=result["pdf"], filetype="pdf") as pdf:
            expected = pdf[1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).tobytes("png")
        self.assertEqual(result["image"], expected)


if __name__ == "__main__":
    unittest.main()
