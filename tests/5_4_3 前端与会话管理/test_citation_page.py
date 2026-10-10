"""引用跳转必须定位实际上传原文的物理页，不能接受任意路径。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hashlib import sha256
import tempfile
import unittest
from unittest.mock import patch
import pymupdf
from src.frontend.components.documents import read_pdf_page
from src.utils.config import load_config, ollama_base_url


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

    def test_rejects_paths_wrong_pages_changed_and_deleted_sources(self):
        for page in (0, 3, True, "2"):
            with self.subTest(page=page), self.assertRaises(ValueError):
                read_pdf_page(self.root, {**self.reference, "metadata": {
                    **self.reference["metadata"], "page_number": page}})
        with self.assertRaises(ValueError):
            read_pdf_page(self.root, {**self.reference, "source_file": "../paper.pdf"})
        self.source.write_bytes(self.data + b"changed")
        with self.assertRaises(ValueError):
            read_pdf_page(self.root, self.reference)
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            read_pdf_page(self.root, self.reference)

    def test_rejects_symlink(self):
        target = self.root / "outside.pdf"
        target.write_bytes(self.data)
        self.source.unlink()
        self.source.symlink_to(target)
        with self.assertRaises(FileNotFoundError):
            read_pdf_page(self.root, self.reference)


class TestContainerLocalAddress(unittest.TestCase):
    def test_container_service_requires_container_and_never_accepts_cloud(self):
        settings = {"provider": "ollama", "base_url": "http://ollama:11434/"}
        with patch("src.utils.config.Path.is_file", return_value=False), self.assertRaises(ValueError):
            ollama_base_url(settings)
        with patch("src.utils.config.Path.is_file", return_value=True):
            self.assertEqual(ollama_base_url(settings), "http://ollama:11434")
            for address in ("https://ollama:11434", "http://example.com", "http://user@ollama:11434",
                            "http://ollama:11434?redirect=remote"):
                with self.subTest(address=address), self.assertRaises(ValueError):
                    ollama_base_url({**settings, "base_url": address})

    def test_environment_override_does_not_modify_yaml(self):
        path = Path(__file__).resolve().parents[2] / "config.yaml"
        before = path.read_bytes()
        with patch.dict("os.environ", {"RAG_OLLAMA_BASE_URL": "http://ollama:11434"}):
            self.assertEqual(load_config()["llm"]["base_url"], "http://ollama:11434")
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
