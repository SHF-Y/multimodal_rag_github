"""
会话管理器单元测试
对会话（Session）和会话管理器（SessionManager）的单元测试。
"""
import pytest
import time
from unittest.mock import MagicMock
from core.session_manager import SessionManager, Session
from langchain_core.messages import HumanMessage, AIMessage

class TestSession:
    def test_add_message(self):
        """测试添加消息"""

        session = Session("test-1")

        msg = HumanMessage(content="你好")
        session.add_message(msg)
        assert len(session.messages) == 1
        assert session.messages[0].content == "你好"

    def test_get_context_messages_with_sliding_window(self):
        """测试滑动窗口只返回最近N条消息"""

        session = Session("test-2", max_history=3)
        for i in range(10):
            session.add_message(HumanMessage(content=f"消息{i}"))

        messages = session.get_context_messages("系统提示")

        assert len(messages) == 4

        assert messages[-1].content == "消息9"

        assert messages[-3].content == "消息7"

    def test_get_context_with_summary(self):
        """测试包含历史摘要的上下文"""
        session = Session("test-3")

        session.summary = "用户询问了变压器的相关问题"

        session.add_message(HumanMessage(content="闪点是多少？"))

        messages = session.get_context_messages("系统提示")

        assert len(messages) == 3

        assert "历史对话摘要" in messages[1].content

    def test_should_summarize_threshold(self):
        """测试摘要压缩阈值判断"""

        session = Session("test-4", summary_threshold=5)

        for i in range(4):
            session.add_message(HumanMessage(content=f"msg{i}"))

        assert not session.should_summarize()

        session.add_message(HumanMessage(content="msg5"))
        session.add_message(HumanMessage(content="msg6"))

        assert session.should_summarize()

class TestSessionManager:

    def test_get_or_create_new_session(self):
        """测试创建新会话"""

        manager = SessionManager()

        session = manager.get_or_create("new-session")

        assert session.session_id == "new-session"

        assert len(session.messages) == 0

    def test_get_existing_session(self):
        """测试获取已存在会话"""

        manager = SessionManager()

        s1 = manager.get_or_create("session-1")

        s1.add_message(HumanMessage(content="test"))

        s2 = manager.get_or_create("session-1")

        assert s2 is s1

        assert len(s2.messages) == 1

    def test_delete_session(self):
        """测试删除会话"""

        manager = SessionManager()

        manager.get_or_create("to-delete")

        manager.delete("to-delete")

        assert "to-delete" not in manager.sessions

    def test_max_sessions_eviction(self):
        """测试超过最大会话数时淘汰最旧的"""

        manager = SessionManager(max_sessions=3)

        for i in range(5):
            manager.get_or_create(f"session-{i}")

            time.sleep(0.01)

        assert len(manager.sessions) <= 3