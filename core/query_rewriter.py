"""
查询改写模块：将用户原始问题改写为更适合检索的形式
支持三种策略：
1. Query Expansion：扩展同义词、专业术语
2. Reference Resolution：消解多轮对话中的指代（"它""这个"）
3. Sub-query Decomposition：将复杂问题拆分为多个子问题
"""
import json
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

QUERY_REWRITE_SYSTEM = """你是一个检索查询优化专家。你的任务是将用户的原始问题改写为更适合向量检索和关键词检索的形式。

规则：
1. 提取问题中的核心实体、专业术语、型号编号
2. 补充可能的同义词、缩写（如"变压器"可扩展为"电力变压器/配电变压器"）
3. 如果问题有指代（如"它""这个""上述设备"），结合历史对话消解
4. 保持改写后的查询简洁，不超过50字
5. 只返回改写后的查询文本，不要解释"""

SUB_QUERY_SYSTEM = """你是一个复杂问题分解专家。将用户的复杂问题分解为2-4个可以独立检索的子问题。

规则：
1. 每个子问题应该是一个完整、独立的问题
2. 子问题之间应该有逻辑递进关系
3. 返回JSON格式：{"sub_queries": ["子问题1", "子问题2", ...]}
4. 只返回JSON，不要其他内容"""

class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def expand(self, query: str, history: Optional[List] = None) -> str:
        """查询扩展：补充同义词和专业术语"""
        history_text = ""
        if history:
            history_text = "\n历史对话：\n" + "\n".join(
                [f"用户：{h['q']}\n助手：{h['a']}" for h in history[-3:]]
            )

        messages = [
            SystemMessage(content=QUERY_REWRITE_SYSTEM),
            HumanMessage(content=f"原始问题：{query}{history_text}\n\n请输出改写后的查询：")
        ]
        result = self.llm.invoke(messages)
        return result.content.strip()

    def decompose(self, query: str) -> List[str]:
        """复杂问题分解：拆分为多个子问题"""
        messages = [
            SystemMessage(content=SUB_QUERY_SYSTEM),
            HumanMessage(content=f"复杂问题：{query}\n\n请分解为子问题：")
        ]
        result = self.llm.invoke(messages)
        try:

            content = result.content.strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
            data = json.loads(content)

            return data.get("sub_queries", [query])
        except Exception:

            return [query]

    def rewrite(self, query: str, history: Optional[List] = None,
                decompose: bool = False) -> List[str]:
        """
        统一改写入口
        Args:
            query: 原始问题
            history: 多轮对话历史
            decompose: 是否需要分解复杂问题
        Returns:
            改写后的查询列表（可能有多个子查询）
        """
        if decompose:
            sub_queries = self.decompose(query)
            return [self.expand(sq, history) for sq in sub_queries]
        return [self.expand(query, history)]