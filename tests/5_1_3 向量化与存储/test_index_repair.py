"""真实Chroma标签损坏注入、重启验证、失败保留和替换回滚；不等同于Windows成因复现。"""

import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from langchain_core.documents import Document
from src.retrieval.repair_index import activate_index, rebuild_index, verify_index
from src.retrieval.vector_store import VectorStore
from tests.helpers import SmallEmbeddings


class TestIndexRepair(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="真实索引修复-")
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "原库"
        self.output = self.source.with_name("新库")

    def worker(self, code, *args, check=True):
        result = subprocess.run([sys.executable, "-c", code, *map(str, args)], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        if check:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_native_label_damage_is_rebuilt_verified_after_restart_and_backed_up(self):
        # 超过sync_threshold使HNSW和映射真实落盘；随后破坏一个自建测试标签。
        self.worker('''import sys
from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStore
from tests.helpers import SmallEmbeddings
s=VectorStore(sys.argv[1], SmallEmbeddings())
s.add_chunks([Document(page_content='农业' if i%2 else '神经网络', metadata={'chunk_id':str(i), 'doc_id':'a' if i%2 else 'b', 'source_file':'论文.pdf','page_number':i%22+1}) for i in range(1002)])
''', self.source)
        metadata_file = next(self.source.glob("*/index_metadata.pickle"))
        with metadata_file.open("rb") as stream:
            data = pickle.load(stream)
        key = next(iter(data.id_to_label))
        old_label = data.id_to_label[key]
        data.id_to_label[key] = 999999
        data.label_to_id[999999] = data.label_to_id.pop(old_label)
        with metadata_file.open("wb") as stream:
            pickle.dump(data, stream)
        damaged = self.worker('''import sys
from src.retrieval.vector_store import VectorStore
s=VectorStore(sys.argv[1]);s._store.get(include=['embeddings'])
''', self.source, check=False)
        self.assertNotEqual(damaged.returncode, 0)
        self.assertIn("Label not found", damaged.stderr)
        self.worker('''import sys
from src.retrieval.repair_index import rebuild_index
from tests.helpers import SmallEmbeddings
rebuild_index(sys.argv[1],sys.argv[2],embeddings=SmallEmbeddings())
''', self.source, self.output)
        self.assertFalse(json.loads((self.output / "repair.json").read_text(encoding="utf-8"))["verified_after_restart"])
        with self.assertRaisesRegex(ValueError, "独立进程"):
            activate_index(self.source, self.output)
        self.worker('''import sys
from src.retrieval.repair_index import verify_index
verify_index(sys.argv[1])
''', self.output)
        backup = Path(activate_index(self.source, self.output))
        self.assertTrue(backup.is_dir())
        self.assertFalse(self.output.exists())
        self.worker('''import sys
from src.retrieval.vector_store import VectorStore
from tests.helpers import SmallEmbeddings
s=VectorStore(sys.argv[1],SmallEmbeddings())
assert s.count()==1002
assert all('retrieval_warning' not in d.metadata for d,_ in s.search('农业',doc_id='a'))
assert s.search('神经网络')[0][1]>0.99
assert not s._store.embeddings.document_calls
''', self.source)
        still_damaged = self.worker('''import sys
from src.retrieval.vector_store import VectorStore
VectorStore(sys.argv[1])._store.get(include=['embeddings'])
''', backup, check=False)
        self.assertIn("Label not found", still_damaged.stderr)

    def test_failed_encoding_keeps_original_and_refuses_existing_output(self):
        store = VectorStore(self.source, SmallEmbeddings())
        chunks = [Document(page_content="原文", metadata={"chunk_id": "1", "doc_id": "d"})]
        store.add_chunks(chunks)
        broken = SmallEmbeddings()
        with patch.object(broken, "embed_documents", side_effect=RuntimeError("模型不可用")):
            with self.assertRaisesRegex(RuntimeError, "模型不可用"):
                rebuild_index(self.source, self.output, embeddings=broken)
        self.assertEqual(store.list_chunks(), chunks)
        with self.assertRaises(FileExistsError):
            rebuild_index(self.source, self.output, embeddings=SmallEmbeddings())
        self.assertTrue(self.source.is_dir())

    def test_changed_original_is_not_replaced_with_stale_snapshot(self):
        store = VectorStore(self.source, SmallEmbeddings())
        store.add_chunks([Document(page_content="原文", metadata={"chunk_id": "1", "doc_id": "d"})])
        rebuild_index(self.source, self.output, embeddings=SmallEmbeddings())
        store.add_chunks([Document(page_content="新增", metadata={"chunk_id": "2", "doc_id": "d"})])
        with self.assertRaisesRegex(ValueError, "发生变化"):
            verify_index(self.output)
        self.assertFalse(json.loads((self.output / "repair.json").read_text(encoding="utf-8"))["verified_after_restart"])
        self.assertEqual(store.count(), 2)

    def test_locked_target_rolls_back_directory_replacement(self):
        self.source.mkdir()
        (self.source / "original.txt").write_text("原库")
        self.output.mkdir()
        (self.output / "repair.json").write_text(json.dumps({"source": str(self.source.resolve()), "verified_after_restart": True}))
        rename = Path.rename
        def locked(path, target):
            if path == self.output.resolve():
                raise PermissionError("Windows文件占用")
            return rename(path, target)
        with patch.object(Path, "rename", locked), self.assertRaises(PermissionError):
            activate_index(self.source, self.output)
        self.assertEqual((self.source / "original.txt").read_text(), "原库")
        self.assertTrue(self.output.is_dir())
        self.assertEqual(list(self.source.parent.glob("原库-backup-*")), [])


if __name__ == "__main__":
    unittest.main()
