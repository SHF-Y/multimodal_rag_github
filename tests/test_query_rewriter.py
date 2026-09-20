"""query_rewriter 单元测试：查询扩展、指代消解、子问题分解"""
from unittest.mock import MagicMock

from core.query_rewriter import QueryRewriter

def make_llm(content):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm

class TestExpand:
    def test_returns_rewritten_query(self):
        llm = make_llm("变压器 闪点 标准")

        qr = QueryRewriter(llm)
        assert qr.expand("变压器闪点多少") == "变压器 闪点 标准"

    def test_history_appended(self):
        llm = make_llm("它=变压器")
        qr = QueryRewriter(llm)
        qr.expand("它闪点多少", history=[{"q": "变压器是什么", "a": "变压器是..."}])

        messages = llm.invoke.call_args[0][0]
        human = [m for m in messages if getattr(m, "type", "") == "human"][-1]

        assert "历史对话" in human.content
        assert "变压器" in human.content

class TestDecompose:
    def test_parses_json(self):
        llm = make_llm('{"sub_queries": ["问题A", "问题B"]}')
        qr = QueryRewriter(llm)
        assert qr.decompose("复杂问题") == ["问题A", "问题B"]

    def test_strips_markdown_block(self):
        llm = make_llm('```json\n{"sub_queries": ["问题X"]}\n```')
        qr = QueryRewriter(llm)
        assert qr.decompose("复杂问题") == ["问题X"]

    def test_falls_back_to_original_on_bad_json(self):
        llm = make_llm("问题1，问题2")
        
        qr = QueryRewriter(llm)
        assert qr.decompose("原问题") == ["原问题"]

class TestRewrite:
    def test_single_query(self):
        llm = make_llm("改写后的查询")
        qr = QueryRewriter(llm)
        assert qr.rewrite("原始问题") == ["改写后的查询"]
        assert llm.invoke.call_count == 1

    def test_decompose_mode_expands_each_subquery(self):
        """decompose=True 时：先分解为子问题，再对每个子问题扩展"""
        llm = MagicMock()
        def fake_invoke(messages):
            human = messages[-1].content
            if "请分解为子问题" in human:
                return MagicMock(content='{"sub_queries": ["q1", "q2"]}')
            return MagicMock(content="扩展:" + human[:20])

        llm.invoke.side_effect = fake_invoke

        qr = QueryRewriter(llm)
        result = qr.rewrite("复杂问题", decompose=True)
        assert len(result) == 2
        assert all(r.startswith("扩展:") for r in result)

        assert llm.invoke.call_count == 3

    def test_default_no_decompose(self):
        llm = make_llm("查询")
        qr = QueryRewriter(llm)
        assert qr.rewrite("问题") == ["查询"]
        assert llm.invoke.call_count == 1
