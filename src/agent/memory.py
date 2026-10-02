"""SQLite会话隔离与本地Qwen历史Token窗口；原始记录保留，摘要后续实现。"""

from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from tokenizers import Tokenizer

from src.utils.config import load_config


def _nonempty(value: str, name: str):
    """身份和消息必须明确传入；不把缺省用户合并到一个共享会话。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}必须是非空字符串")


@lru_cache(maxsize=1)
def _load_tokenizer(path: str):
    """只缓存本地词表，不缓存任何会话；计数必须关闭分词器自带的裁剪/填充。"""
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def count_history_tokens(history: list[dict]) -> int:
    """统计与Agent一致的历史JSON（含角色/分隔符），不是整个Ollama请求的用量。"""
    config = load_config()
    model = config["llm"]["model"]
    if not isinstance(model, str) or not model.startswith("qwen2.5:"):
        raise ValueError("当前历史分词器仅适用于Qwen2.5，更换生成模型需适配对应分词器")
    path = Path(config["memory"]["tokenizer_path"]).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.is_file():
        raise FileNotFoundError(f"缺少本地Qwen分词器：{path}；请先准备分词器，不会自动联网下载")
    tokenizer = _load_tokenizer(str(path))
    if not history:
        return 0  # 空历史的JSON框架属于基础请求，不占历史内容预算。
    text = json.dumps(history, ensure_ascii=False, allow_nan=False)
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


class MemoryManager:
    """沿用上游SQLite/Message思路，仅保留当前需要的会话操作。

    每次操作单独连接，不跨线程复用连接，不缓存其他会话的消息。
    user_id应由可信调用方确定；归属校验不等同于用户登录认证。
    """

    def __init__(self, db_path: str | Path | None = None):
        path = db_path if db_path is not None else load_config()["paths"]["session_db"]
        self.db_path = Path(path).expanduser()
        if not self.db_path.is_absolute():
            self.db_path = Path(__file__).resolve().parents[2] / self.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    role TEXT NOT NULL CHECK(role IN ('human', 'ai')),
                    content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            """)

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _check_session(connection, user_id: str, session_id: str):
        _nonempty(user_id, "user_id")
        _nonempty(session_id, "session_id")
        row = connection.execute("SELECT user_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise LookupError("会话不存在，请先创建会话")
        if row[0] != user_id:
            raise PermissionError("不能访问其他用户的会话")

    def create_session(self, user_id: str) -> str:
        """创建归属固定的独立会话，UUID只用于标识，不用作登录凭据。"""
        _nonempty(user_id, "user_id")
        session_id = uuid4().hex
        with closing(self._connect()) as connection, connection:
            connection.execute("INSERT INTO sessions(session_id, user_id) VALUES(?, ?)", (session_id, user_id))
        return session_id

    def list_sessions(self, user_id: str) -> list[str]:
        """只列出本用户的会话，包括尚未产生消息的空会话。"""
        _nonempty(user_id, "user_id")
        with closing(self._connect()) as connection:
            return [row[0] for row in connection.execute(
                "SELECT session_id FROM sessions WHERE user_id=? ORDER BY rowid", (user_id,))]

    def get_messages(self, user_id: str, session_id: str) -> list:
        """按入库顺序返回新的LangChain消息对象，修改返回值不会污染存储。"""
        with closing(self._connect()) as connection:
            self._check_session(connection, user_id, session_id)
            rows = connection.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
                                      (session_id,)).fetchall()
        classes = {"human": HumanMessage, "ai": AIMessage}
        return [classes[role](content=content) for role, content in rows]

    def append_turn(self, user_id: str, session_id: str, question: str, answer: str):
        """事务内追加一整轮，避免覆盖其他线程追加的历史或只保存半轮。"""
        _nonempty(question, "question")
        _nonempty(answer, "answer")
        with closing(self._connect()) as connection, connection:
            self._check_session(connection, user_id, session_id)
            connection.executemany("INSERT INTO messages(session_id, role, content) VALUES(?, ?, ?)",
                                   [(session_id, "human", question), (session_id, "ai", answer)])

    def get_context(self, user_id: str, session_id: str) -> dict:
        """按完整问答对移除最旧历史，保留连续的最近窗口，不删除数据库原文。"""
        messages = self.get_messages(user_id, session_id)  # 先校验归属，不能裁剪/读取外用户历史。
        limit = load_config()["memory"]["max_history_tokens"]
        if type(limit) is not int or limit < 1:
            raise ValueError("memory.max_history_tokens必须为正整数")
        history = [{"role": message.type, "content": message.content} for message in messages]
        original_turns = len(history) // 2
        tokens = count_history_tokens(history)
        while history and tokens > limit:
            del history[:2]  # 一整轮一起移除；单轮超限时也不留下孤立的回答或截断引用。
            tokens = count_history_tokens(history)  # BPE计数不简单相加，按真实后缀重新统计。
        return {"history": history, "history_window": {"tokens": tokens, "max_tokens": limit,
                "retained_turns": len(history) // 2, "dropped_turns": original_turns - len(history) // 2,
                "tokenizer": "Qwen2.5"}}

    def clear_session(self, user_id: str, session_id: str):
        """只清空已归属会话的历史；保留会话标识，不触及其他会话。"""
        with closing(self._connect()) as connection, connection:
            self._check_session(connection, user_id, session_id)
            connection.execute("DELETE FROM messages WHERE session_id=?", (session_id,))


def run_session(question: str, user_id: str, session_id: str, tools=None, *, memory: MemoryManager | None = None):
    """有记忆的Agent入口：先校验归属，只从当前会话重建历史Context。

    不接收外部执行Context，避免误传其他会话的工具结果或恢复状态。
    完整消费到done后才保存用户问题/最终回答；关闭生成器不留下半轮。
    最终错误提示也属于对话历史，但不把中间工具轨迹或禁用集合带到下一问题。
    """
    from src.agent.react_loop import run_react

    _nonempty(question, "question")
    memory = memory if memory is not None else MemoryManager()
    context = memory.get_context(user_id, session_id)
    for event in run_react(question, tools, context):
        if event["type"] == "done":
            memory.append_turn(user_id, session_id, question, event["full_response"])
        yield {**event, "user_id": user_id, "session_id": session_id}
