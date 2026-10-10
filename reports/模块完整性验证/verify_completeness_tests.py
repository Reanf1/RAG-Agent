"""模块四专项与全项目回归入口；记录实际测试名称、失败和跳过，不覆盖历史证据。"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 按实际职责选取模块四用例，不把全部检索或Agent算法测试算作前端测试。
MODULE_FOUR_CLASSES = {
    "TestImportFrontend", "TestDocumentManagement", "TestHealthCheck", "TestHealthCheckPage",
    "TestStreamingFrontend", "TestSessionIsolation",
    "TestRAGSearchRouting", "TestAgentMemoryContext", "TestAgentMetrics",
    "TestAgentTraceMetrics", "TestAgentMetricsEntryAndPage",
    "TestCitationPage", "TestContainerLocalAddress",
}


def flatten(suite):
    """展开unittest套件，保留每个用例的原始标识。"""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["module4", "all"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("证据文件已存在，请使用新的输出路径")
    args.output = args.output.resolve()
    if sys.platform == "win32":
        # 相对路径样例与TEMP保持同一盘符，Windows不能计算C盘到E盘的relpath。
        os.chdir(tempfile.gettempdir())
    loader = unittest.TestLoader()
    tests = list(flatten(loader.discover(str(ROOT / "tests"), pattern="test_*.py")))
    if args.scope == "module4":
        tests = [test for test in tests if type(test).__name__ in MODULE_FOUR_CLASSES]
    identities = [test.id() for test in tests]
    started = perf_counter()
    original_cleanup = tempfile.TemporaryDirectory.cleanup
    def cleanup_test_directory(directory):
        """测试结束先释放该临时目录的Chroma句柄，不停止其他索引或忽略删除错误。"""
        from chromadb.api.client import SharedSystemClient
        root = Path(directory.name).resolve()
        for identifier, system in list(SharedSystemClient._identifier_to_system.items()):
            path = Path(system.settings.persist_directory).resolve()
            if path.is_relative_to(root):
                system.stop()
                SharedSystemClient._identifier_to_system.pop(identifier, None)
        original_cleanup(directory)
    with patch.object(tempfile.TemporaryDirectory, "cleanup", cleanup_test_directory):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
    report = {
        "scope": args.scope, "elapsed_seconds": perf_counter() - started,
        "run": result.testsRun, "classes": dict(Counter(type(test).__name__ for test in tests)),
        "tests": identities, "failures": [(t.id(), error) for t, error in result.failures],
        "errors": [(t.id(), error) for t, error in result.errors],
        "skipped": [(t.id(), reason) for t, reason in result.skipped],
        "unexpected_successes": [t.id() for t in result.unexpectedSuccesses],
        "expected_failures": [(t.id(), error) for t, error in result.expectedFailures],
        "passed": result.wasSuccessful() and not result.skipped and not result.expectedFailures,
        "boundary": "模型HTTP或小型向量隔离见各用例注释；自动化通过不代表真实答案质量全通过。",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("scope", "run", "classes", "elapsed_seconds", "passed")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
