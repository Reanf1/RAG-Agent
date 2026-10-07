"""5.3.4 多轮对话记忆管理：TestSessionIsolation。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
import json
import sqlite3
import subprocess
import tempfile
from threading import Barrier, Lock
import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage
from src.agent.memory import MemoryManager, run_session
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestSessionIsolation(unittest.TestCase):
    """真实临时SQLite隔离测试；模型HTTP按既有方式隔离，不替换存储和Agent循环。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sessions" / "memory.sqlite3"
        self.memory = MemoryManager(self.path)
        self.a1 = self.memory.create_session("alice")
        self.a2 = self.memory.create_session("alice")
        self.b1 = self.memory.create_session("bob")

    def contents(self, user, session):
        return [message.content for message in self.memory.get_messages(user, session)]

    def packet(self, content):
        return {"model": "qwen2.5:7b", "message": {"content": json.dumps(content, ensure_ascii=False)},
                "done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 20}

    def packets(self, answer="回答", complete=True):
        return [self.packet({"thought": "结合本会话已有历史回答。", "next_step": "answer", "tool_name": None}),
                self.packet({"observation": "已有信息足够回答。", "decision": "finish",
                             "task_complete": complete, "answer": answer})]

    def run_turn(self, user, session, question="追问", answer="回答", complete=True):
        with patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(p).encode()) for p in self.packets(answer, complete)]) as http:
            events = list(run_session(question, user, session, tools=[], memory=self.memory))
        return events, http

    def test_new_sessions_are_unique_empty_and_owned_lists_include_no_foreign_session(self):
        self.assertEqual(len({self.a1, self.a2, self.b1}), 3)
        self.assertEqual(self.contents("alice", self.a1), [])
        self.assertEqual(self.memory.list_sessions("alice"), [self.a1, self.a2])
        self.assertEqual(self.memory.list_sessions("bob"), [self.b1])
        self.assertEqual(self.memory.list_sessions("new-user"), [])

    def test_same_user_different_sessions_have_independent_history(self):
        self.memory.append_turn("alice", self.a1, "A的问题", "A的回答")
        self.memory.append_turn("alice", self.a2, "B的问题", "B的回答")
        self.assertEqual(self.contents("alice", self.a1), ["A的问题", "A的回答"])
        self.assertEqual(self.contents("alice", self.a2), ["B的问题", "B的回答"])

    def test_different_users_keep_independent_history(self):
        self.memory.append_turn("alice", self.a1, "Alice资料", "Alice回答")
        self.memory.append_turn("bob", self.b1, "Bob资料", "Bob回答")
        self.assertEqual(self.contents("alice", self.a1), ["Alice资料", "Alice回答"])
        self.assertEqual(self.contents("bob", self.b1), ["Bob资料", "Bob回答"])

    def test_foreign_session_read_append_and_clear_are_rejected_without_changes(self):
        self.memory.append_turn("alice", self.a1, "私有问题", "私有回答")
        for operation in (lambda: self.memory.get_messages("bob", self.a1),
                          lambda: self.memory.append_turn("bob", self.a1, "覆盖", "覆盖"),
                          lambda: self.memory.clear_session("bob", self.a1)):
            with self.assertRaises(PermissionError):
                operation()
        self.assertEqual(self.contents("alice", self.a1), ["私有问题", "私有回答"])
        self.assertEqual(self.contents("bob", self.b1), [])

    def test_unknown_session_is_not_implicitly_created(self):
        for operation in (lambda: self.memory.get_messages("alice", "missing"),
                          lambda: self.memory.append_turn("alice", "missing", "问题", "回答"),
                          lambda: self.memory.clear_session("alice", "missing")):
            with self.assertRaises(LookupError):
                operation()
        self.assertEqual(self.memory.list_sessions("alice"), [self.a1, self.a2])

    def test_messages_keep_role_order_unicode_formula_and_markdown(self):
        for index in range(3):
            self.memory.append_turn("alice", self.a1, f"问题{index}: α² **Transformer**", f"回答{index}: $x_1$ 中文")
        messages = self.memory.get_messages("alice", self.a1)
        self.assertEqual([message.type for message in messages], ["human", "ai"] * 3)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(messages[-2].content, "问题2: α² **Transformer**")
        self.assertEqual(messages[-1].content, "回答2: $x_1$ 中文")

    def test_returned_message_mutation_does_not_modify_store_or_other_session(self):
        self.memory.append_turn("alice", self.a1, "原问题", "原回答")
        messages = self.memory.get_messages("alice", self.a1)
        messages[0].content = "更改返回对象"
        messages.append(HumanMessage(content="不应写回"))
        self.assertEqual(self.contents("alice", self.a1), ["原问题", "原回答"])
        self.assertEqual(self.contents("alice", self.a2), [])

    def test_reopening_manager_restores_both_users_without_shared_message_cache(self):
        self.memory.append_turn("alice", self.a1, "持久问题A", "持久回答A")
        self.memory.append_turn("bob", self.b1, "持久问题B", "持久回答B")
        reopened = MemoryManager(self.path)
        self.assertEqual(reopened.get_messages("alice", self.a1)[0].content, "持久问题A")
        self.assertEqual(reopened.get_messages("bob", self.b1)[0].content, "持久问题B")
        reopened.append_turn("alice", self.a1, "追加", "答复")
        self.assertEqual(len(self.memory.get_messages("alice", self.a1)), 4)

    def test_new_python_process_restores_persistent_history(self):
        self.memory.append_turn("alice", self.a1, "独立进程恢复问题", "独立进程恢复回答")
        script = ("import json,sys; from src.agent.memory import MemoryManager; "
                  "print(json.dumps([m.content for m in MemoryManager(sys.argv[1]).get_messages(sys.argv[2],sys.argv[3])]))")
        result = subprocess.run([sys.executable, "-c", script, str(self.path), "alice", self.a1],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                                check=True, timeout=15)
        self.assertEqual(json.loads(result.stdout), ["独立进程恢复问题", "独立进程恢复回答"])

    def test_clear_only_current_owned_session_and_keep_identifier(self):
        for user, session in (("alice", self.a1), ("alice", self.a2), ("bob", self.b1)):
            self.memory.append_turn(user, session, session, "回答")
        self.memory.clear_session("alice", self.a1)
        self.assertEqual(self.contents("alice", self.a1), [])
        self.assertEqual(len(self.contents("alice", self.a2)), 2)
        self.assertEqual(len(self.contents("bob", self.b1)), 2)
        self.assertIn(self.a1, self.memory.list_sessions("alice"))

    def test_invalid_identity_or_message_does_not_save_half_turn(self):
        for value in (None, "", "  ", 123):
            with self.subTest(value=value):
                for operation in (lambda: self.memory.create_session(value),
                                  lambda: self.memory.list_sessions(value),
                                  lambda: self.memory.get_messages(value, self.a1),
                                  lambda: self.memory.get_messages("alice", value),
                                  lambda: self.memory.append_turn("alice", self.a1, value, "回答"),
                                  lambda: self.memory.append_turn("alice", self.a1, "问题", value)):
                    with self.assertRaises(ValueError):
                        operation()
        self.assertEqual(self.contents("alice", self.a1), [])

    def test_sql_metacharacters_are_literal_identity_values(self):
        user = "alice' OR 1=1 --"
        session = self.memory.create_session(user)
        self.memory.append_turn(user, session, "自己的问题", "自己的回答")
        self.assertEqual(self.memory.list_sessions(user), [session])
        with self.assertRaises(PermissionError):
            self.memory.get_messages(user, self.a1)
        with self.assertRaises(LookupError):
            self.memory.get_messages("alice", "' OR 1=1 --")

    def test_second_insert_failure_rolls_back_entire_turn(self):
        self.memory.append_turn("alice", self.a1, "已有问题", "已有回答")
        with sqlite3.connect(self.path) as connection:
            connection.executescript("""CREATE TRIGGER fail_answer BEFORE INSERT ON messages
                WHEN NEW.content='FAIL' BEGIN SELECT RAISE(ABORT, '明确注入的写入失败'); END;""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.append_turn("alice", self.a1, "不应残留的问题", "FAIL")
        self.assertEqual(self.contents("alice", self.a1), ["已有问题", "已有回答"])

    def test_threads_and_manager_instances_append_owned_sessions_without_cross_contamination(self):
        sessions = [(f"user-{index % 2}", self.memory.create_session(f"user-{index % 2}")) for index in range(8)]
        barrier = Barrier(4)
        managers = [self.memory, MemoryManager(self.path)]
        def write(index):
            user, session = sessions[index]
            barrier.wait(timeout=3)
            for turn in range(4):
                managers[index % 2].append_turn(user, session, f"session-{index}-q-{turn}", f"session-{index}-a-{turn}")
            return [message.content for message in managers[index % 2].get_messages(user, session)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            histories = list(pool.map(write, range(8)))
        for index, history in enumerate(histories):
            self.assertEqual(history, [value for turn in range(4) for value in
                                     (f"session-{index}-q-{turn}", f"session-{index}-a-{turn}")])

    def test_same_session_concurrent_appends_do_not_overwrite_or_split_turns(self):
        barrier = Barrier(4)
        def write(index):
            barrier.wait(timeout=3)
            for turn in range(5):
                self.memory.append_turn("alice", self.a1, f"{index}:{turn}", f"answer:{index}:{turn}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write, range(4)))
        history = self.contents("alice", self.a1)
        self.assertEqual(len(history), 40)
        self.assertEqual(len(set(history[::2])), 20)
        for question, answer in zip(history[::2], history[1::2]):
            self.assertEqual(answer, "answer:" + question)

    def test_default_manager_uses_configured_path(self):
        config = deepcopy(load_config())
        config["paths"]["session_db"] = str(Path(self.directory.name) / "configured" / "history.sqlite3")
        with patch("src.agent.memory.load_config", return_value=config):
            memory = MemoryManager()
        self.assertTrue(memory.db_path.is_file())
        self.assertEqual(memory.db_path, Path(config["paths"]["session_db"]))

    def test_agent_requests_only_current_session_history_and_saves_finished_pair(self):
        self.memory.append_turn("alice", self.a1, "我的研究代号是ALICE_A_ONLY。", "已记录ALICE_A_ONLY。")
        self.memory.append_turn("alice", self.a2, "我的研究代号是ALICE_B_ONLY。", "已记录ALICE_B_ONLY。")
        self.memory.append_turn("bob", self.b1, "我的研究代号是BOB_ONLY。", "已记录BOB_ONLY。")
        events, http = self.run_turn("alice", self.a1, answer="你的代号是ALICE_A_ONLY。")
        self.assertTrue(events[-1]["task_complete"])
        for call in http.call_args_list:
            body = call.args[0].data.decode()
            self.assertIn("ALICE_A_ONLY", body)
            self.assertNotIn("ALICE_B_ONLY", body)
            self.assertNotIn("BOB_ONLY", body)
        self.assertTrue(all(e["user_id"] == "alice" and e["session_id"] == self.a1 for e in events))
        self.assertEqual(self.contents("alice", self.a1)[-2:], ["追问", "你的代号是ALICE_A_ONLY。"])
        self.assertEqual(len(self.contents("alice", self.a2)), 2)
        self.assertEqual(len(self.contents("bob", self.b1)), 2)

    def test_agent_foreign_access_fails_before_any_model_or_tool_call(self):
        with patch("src.agent.react_loop.run_react") as run_mock, self.assertRaises(PermissionError):
            list(run_session("读取资料", "bob", self.a1, memory=self.memory))
        run_mock.assert_not_called()
        self.assertEqual(self.contents("alice", self.a1), [])

    def test_concurrent_agent_requests_keep_model_context_history_and_logs_owned(self):
        """真实Agent/SQLite/日志并发；仅模型HTTP构造，屏障要求两请求同时到达。"""
        identities = [("alice", self.a1, "ALICE_CONCURRENT_ONLY"),
                      ("bob", self.b1, "BOB_CONCURRENT_ONLY")]
        for user, session, code in identities:
            self.memory.append_turn(user, session, f"我的代号是{code}。", "已记录。")
        config = deepcopy(load_config())
        log_dir = Path(self.directory.name) / "logs"
        config["paths"]["logs"] = str(log_dir)
        barrier, requests, lock = Barrier(2), [], Lock()
        def reply(request, **kwargs):
            body = json.loads(request.data)
            text = json.dumps(body, ensure_ascii=False)
            codes = [code for _, _, code in identities if code in text]
            self.assertEqual(len(codes), 1)
            with lock:
                requests.append(codes[0])
            # 每个用户的Thought及Observation都等待另一用户，不能用串行执行冒充并发。
            barrier.wait(timeout=5)
            planning = "thought" in body["format"]["properties"]
            content = {"thought": "依据本会话历史回答。", "next_step": "answer", "tool_name": None} if planning else {
                "observation": "本会话记录包含代号。", "decision": "finish", "task_complete": True, "answer": codes[0]}
            return BytesIO(json.dumps(self.packet(content), ensure_ascii=False).encode())
        with patch("src.utils.logger.load_config", return_value=config), \
                patch("src.agent.react_loop.urlopen", side_effect=reply), ThreadPoolExecutor(max_workers=2) as pool:
            streams = list(pool.map(lambda identity: list(run_session("我的代号是什么？", identity[0], identity[1],
                                                                  tools=[], memory=self.memory)), identities))
        request_ids = set()
        for (user, session, code), events in zip(identities, streams):
            self.assertTrue(events[-1]["task_complete"])
            self.assertEqual(events[-1]["full_response"], code)
            request_ids.add(events[-1]["request_id"])
            self.assertTrue(all(e["user_id"] == user and e["session_id"] == session and not e.get("log_error") for e in events))
            self.assertEqual(self.contents(user, session), [f"我的代号是{code}。", "已记录。", "我的代号是什么？", code])
        self.assertEqual(len(request_ids), 2)
        self.assertEqual(sorted(requests), sorted([code for _, _, code in identities] * 2))
        logs = [json.loads(line) for path in log_dir.glob("agent_*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(logs), sum(len(events) for events in streams))
        for (user, session, _), events in zip(identities, streams):
            owned = [row for row in logs if row["request_id"] == events[-1]["request_id"]]
            self.assertEqual([row["event"] for row in owned], [event["type"] for event in events])
            self.assertTrue(all(row["user_id"] == user and row["session_id"] == session for row in owned))
            self.assertEqual(owned[-1]["metrics"]["tokens"]["total"], events[-1]["metrics"]["tokens"]["total"])

    def test_agent_request_context_is_fresh_and_does_not_reuse_prior_tool_state(self):
        self.memory.append_turn("alice", self.a1, "已有问题", "已有回答")
        for _ in range(2):
            _, http = self.run_turn("alice", self.a1)
            body = json.loads(http.call_args_list[0].args[0].data)
            context = json.loads(body["messages"][1]["content"])["context"]
            self.assertEqual(context["observations"], [])
            self.assertNotIn("recovery", context)
            self.assertNotIn("last_observation", context)
            self.assertTrue(context["history"])

    def test_closing_agent_stream_saves_no_incomplete_turn(self):
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packets()[0]).encode())):
            stream = run_session("问题", "alice", self.a1, tools=[], memory=self.memory)
            self.assertEqual(next(stream)["type"], "thought")
            stream.close()
        self.assertEqual(self.contents("alice", self.a1), [])

    def test_incomplete_answer_is_saved_as_real_response_without_claiming_success(self):
        events, _ = self.run_turn("alice", self.a1, answer="资料不足，请提供原文。", complete=False)
        self.assertFalse(events[-1]["task_complete"])
        self.assertEqual(self.contents("alice", self.a1), ["追问", "资料不足，请提供原文。"])

    def test_agent_persistence_failure_does_not_emit_successful_done_or_save_half_turn(self):
        with sqlite3.connect(self.path) as connection:
            connection.executescript("""CREATE TRIGGER fail_all_answers BEFORE INSERT ON messages
                WHEN NEW.role='ai' BEGIN SELECT RAISE(ABORT, '明确注入的保存故障'); END;""")
        with patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(p).encode()) for p in self.packets()]):
            stream = run_session("问题", "alice", self.a1, tools=[], memory=self.memory)
            received = []
            with self.assertRaises(sqlite3.IntegrityError):
                for event in stream:
                    received.append(event)
        self.assertNotIn("done", [event["type"] for event in received])
        self.assertEqual(self.contents("alice", self.a1), [])


if __name__ == "__main__":
    unittest.main()
