"""
会话管理器：管理多轮对话历史
1. 每个会话有唯一session_id，存储对话消息列表
2. 支持滑动窗口：保留最近N轮对话，避免上下文过长
3. 支持摘要压缩：当历史超过阈值时，用LLM生成摘要替代早期对话
4. 线程安全：多线程并发访问同一会话时加锁
"""

import threading
import time
from typing import List, Dict, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
class Session:
    """单个会话的状态"""
    def __init__(self, session_id: str, max_history: int = 10,summary_threshold: int = 15):

        self.session_id = session_id

        self.messages: List[BaseMessage] = []

        self.max_history = max_history

        self.summary_threshold = summary_threshold 

        self.summary: str = ""
        self.created_at = time.time()
        self.last_active = time.time()

        self.lock = threading.Lock()

    def add_message(self, message: BaseMessage):
        """添加一条消息"""

        with self.lock:

            self.messages.append(message)

            self.last_active = time.time()

    def get_context_messages(self, system_prompt: str) -> List[BaseMessage]:
        """
        获取用于LLM输入的消息列表
        策略：系统提示 + 历史摘要（如有） + 最近N轮对话
        """

        with self.lock:

            result = [SystemMessage(content=system_prompt)]

            if self.summary:
                result.append(SystemMessage(
                    content=f"【历史对话摘要】\n{self.summary}"
                ))

            recent = self.messages[-self.max_history:] if self.messages else []

            result.extend(recent)

            return result

    def should_summarize(self) -> bool:
        """判断是否需要触发摘要压缩"""

        return len(self.messages) > self.summary_threshold

    def compress_history(self, llm):
        """
        用LLM对早期对话做摘要压缩；
        保留最近N轮，将更早的对话压缩为摘要；
        【性能优化】：将LLM调用移到锁外执行，避免长时间阻塞其他线程访问会话
        """

        with self.lock:
            if len(self.messages) <= self.summary_threshold:
                return

            early = self.messages[:-self.max_history]

            early_text = "\n".join([
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}：{m.content}"
                for m in early if m.content
            ])

        summary_prompt = f"""请将以下对话历史压缩为一段简洁的摘要（不超过200字），保留关键信息、用户需求和已确认的结论：
{early_text}

摘要："""

        summary_msg = llm.invoke([HumanMessage(content=summary_prompt)])
        new_summary = summary_msg.content.strip()

        with self.lock:

            if len(self.messages) > self.max_history:
                self.messages = self.messages[-self.max_history:]

            self.summary = new_summary

            print(f"会话 {self.session_id} 已压缩历史，摘要长度: {len(new_summary)}")

class SessionManager:
    """全局会话管理器"""
    def __init__(self, max_sessions: int = 1000, session_timeout: int = 3600):

        self.sessions: Dict[str, Session] = {}

        self.max_sessions = max_sessions

        self.session_timeout = session_timeout

        self.lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        """获取或创建会话"""

        with self.lock:

            self._cleanup_expired()

            if session_id not in self.sessions:

                if len(self.sessions) >= self.max_sessions:
                    oldest_id = min(
                        self.sessions.keys(),
                        key=lambda sid: self.sessions[sid].last_active
                    )
                    del self.sessions[oldest_id]

                self.sessions[session_id] = Session(session_id)

            return self.sessions[session_id]

    def _cleanup_expired(self):
        """清理过期会话（调用方需持有锁）"""

        now = time.time()

        expired = [
            sid for sid, s in self.sessions.items()
            if now - s.last_active > self.session_timeout
        ]

        for sid in expired:
            del self.sessions[sid]

    def delete(self, session_id: str):
        """删除会话"""

        with self.lock:

            self.sessions.pop(session_id, None)
