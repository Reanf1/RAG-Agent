"""侧栏历史会话管理；SQLite保存正文、引用和公开轨迹，页面只重绘。"""

from pathlib import Path
import re
from uuid import uuid4

import streamlit as st

from src.agent.memory import MemoryManager


def activate_session(memory: MemoryManager, user_id: str, session_id: str, cancel_request):
    """先校验归属并读取历史，再切换，避免把旧缓存/候选内容送入新会话。"""
    messages = memory.get_messages(user_id, session_id)
    rag_messages = memory.get_rag_messages(user_id, session_id)
    if "rag_pending" in st.session_state:
        cancel_request(st.session_state.rag_pending["message"], "cancelled")
    st.session_state.pop("rag_pending", None)
    if "rag_cache" in st.session_state:
        st.session_state.rag_cache.clear()
    st.session_state.pop("agent_last_event", None)
    st.session_state.agent_question = ""  # 显式通知浏览器清空表单，避免保留另一会话的未提交草稿。
    st.session_state.pop("delete_session_pending", None)
    history = []
    for question, answer in zip(messages[::2], messages[1::2]):
        details = answer.additional_kwargs
        history.append({"question": question.content, "answer": answer.content,
                        "complete": details.get("task_complete"), "stop_reason": details.get("stop_reason")})
    # 旧数据库正文照常显示；没有保存过的完成状态/轨迹不能凭空补齐。
    if messages and messages[-1].additional_kwargs.get("event"):
        st.session_state.agent_last_event = {**messages[-1].additional_kwargs["event"],
                                             "full_response": messages[-1].content, "history_replayed": True}
    st.session_state.agent_messages = history
    st.session_state.rag_messages = rag_messages
    st.session_state.agent_session_id = st.session_state.rag_session_id = session_id
    st.session_state.reset_session_selector = True
    st.query_params["conversation"] = session_id


def render_sessions(db_path: Path, cancel_request) -> bool:
    """管理当前本机访客的会话；随机地址标识用于演示隔离，不是登录认证。"""
    with st.sidebar:
        st.subheader("对话历史管理")
        st.caption("新建与切换同时作用于两个问答标签。保留当前地址可在刷新后找回历史；访客标识不是登录认证。")
        try:
            if "agent_user_id" not in st.session_state:
                visitor = st.query_params.get("visitor", "")
                st.session_state.agent_user_id = visitor if re.fullmatch(r"[0-9a-f]{32}", visitor) else uuid4().hex
            user_id = st.session_state.agent_user_id
            st.query_params["visitor"] = user_id
            if "agent_memory" not in st.session_state:
                st.session_state.agent_memory = MemoryManager(db_path)
            memory = st.session_state.agent_memory
            sessions = memory.list_sessions(user_id)
            requested = st.query_params.get("conversation") or st.session_state.get("agent_session_id")
            current = requested if requested in sessions else (sessions[-1] if sessions else memory.create_session(user_id))
            if current != st.session_state.get("agent_session_id"):
                activate_session(memory, user_id, current, cancel_request)
            if st.button("新建会话", key="new_conversation"):
                activate_session(memory, user_id, memory.create_session(user_id), cancel_request)
                st.rerun()
            sessions = memory.list_sessions(user_id)
            labels = {}
            for identifier in sessions:
                saved = memory.get_messages(user_id, identifier)
                rag_saved = memory.get_rag_messages(user_id, identifier)
                title = saved[0].content if saved else (rag_saved[0]["question"] if rag_saved else "新会话")
                labels[identifier] = f"{title[:24]} · {identifier[:8]}"
            if st.session_state.pop("reset_session_selector", False) or st.session_state.get("conversation_select") not in sessions:
                st.session_state.conversation_select = current
            selected = st.selectbox("历史会话", sessions, key="conversation_select", format_func=labels.get)
            if selected != current:
                activate_session(memory, user_id, selected, cancel_request)
                current = selected
            st.caption(f"当前会话：{current}")
            if st.button("删除当前会话", key="delete_conversation"):
                st.session_state.delete_session_pending = current
            if st.session_state.get("delete_session_pending") == current:
                st.warning(f"待删除：{labels[current]}。问答、引用和摘要移入会话回收区，知识库文档与请求日志保留。")
                if st.button("确认删除会话", key="confirm_delete_conversation"):
                    memory.delete_session(user_id, current)
                    remaining = memory.list_sessions(user_id)
                    activate_session(memory, user_id, remaining[-1] if remaining else memory.create_session(user_id), cancel_request)
                    st.rerun()
                if st.button("取消删除会话", key="cancel_delete_conversation"):
                    st.session_state.pop("delete_session_pending")
                    st.rerun()
            archived = memory.list_sessions(user_id, archived=True)
            if archived:
                restore_id = st.selectbox("会话回收区", archived, key="restore_conversation_select")
                if st.button("恢复会话", key="restore_conversation"):
                    memory.restore_session(user_id, restore_id)
                    activate_session(memory, user_id, restore_id, cancel_request)
                    st.rerun()
            return True
        except Exception as error:
            st.error(f"会话管理失败：{type(error).__name__}: {error}。请检查本地会话数据库后重试。")
            return False
