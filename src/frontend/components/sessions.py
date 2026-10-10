"""侧栏历史会话管理；SQLite保存正文、引用和公开轨迹，页面只重绘。"""

from pathlib import Path
import re
from uuid import uuid4

import streamlit as st

from src.agent.memory import MemoryManager


def activate_session(memory: MemoryManager, user_id: str, session_id: str):
    """先校验归属并读取历史，再切换消息与指标。"""
    messages = memory.get_messages(user_id, session_id)
    st.session_state.pop("agent_last_event", None)
    st.session_state.pop("delete_session_pending", None)
    history = []
    for question, answer in zip(messages[::2], messages[1::2]):
        details = answer.additional_kwargs
        history.append({"question": question.content, "answer": answer.content,
                        "complete": details.get("task_complete"), "stop_reason": details.get("stop_reason"),
                        "event": details.get("event", {})})
    # 旧数据库正文照常显示；没有保存过的完成状态/轨迹不能凭空补齐。
    if messages and messages[-1].additional_kwargs.get("event"):
        st.session_state.agent_last_event = {**messages[-1].additional_kwargs["event"],
                                             "full_response": messages[-1].content, "history_replayed": True}
    st.session_state.agent_messages = history
    st.session_state.agent_session_id = session_id
    st.query_params["conversation"] = session_id


def render_sessions(db_path: Path) -> bool:
    """管理当前本机访客的会话；随机地址标识用于演示隔离，不是登录认证。"""
    with st.sidebar:
        st.header("对话历史管理")
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
                activate_session(memory, user_id, current)
            if st.button("新建会话", key="new_conversation"):
                activate_session(memory, user_id, memory.create_session(user_id))
                st.rerun()
            sessions = memory.list_sessions(user_id)
            labels = {identifier: memory.get_session_title(user_id, identifier) for identifier in sessions}
            # 直接点击会话标题切换；当前会话使用主按钮高亮，ID只用于稳定key。
            with st.container(key="conversation_history"):
                st.caption("最近")
                for identifier in reversed(sessions):
                    if st.button(labels[identifier], key=f"conversation:{identifier}",
                                 type="primary" if identifier == current else "secondary", width="stretch"):
                        if identifier != current:
                            activate_session(memory, user_id, identifier)
                            st.rerun()
            if st.session_state.get("delete_session_pending") == current:
                confirm, cancel = st.columns(2)
                if confirm.button("确认删除", key="confirm_delete_conversation"):
                    memory.delete_session(user_id, current)
                    remaining = memory.list_sessions(user_id)
                    activate_session(memory, user_id, remaining[-1] if remaining else memory.create_session(user_id))
                    st.rerun()
                if cancel.button("取消", key="cancel_delete_conversation"):
                    st.session_state.pop("delete_session_pending")
                    st.rerun()
            elif st.button("删除当前会话", key="delete_conversation"):
                st.session_state.delete_session_pending = current
                st.rerun()
            archived = memory.list_sessions(user_id, archived=True)
            if archived:
                archive_titles = {identifier: memory.get_session_title(user_id, identifier) for identifier in archived}
                restore_id = st.radio("恢复会话", archived, key="restore_conversation_select", width="stretch",
                                      format_func=lambda identifier: archive_titles[identifier],
                                      captions=[f"ID：{identifier[:8]}" for identifier in archived])
                if st.button("恢复", key="restore_conversation"):
                    memory.restore_session(user_id, restore_id)
                    activate_session(memory, user_id, restore_id)
                    st.rerun()
            return True
        except Exception as error:
            st.error(f"会话管理失败：{type(error).__name__}: {error}。请检查本地会话数据库后重试。")
            return False
