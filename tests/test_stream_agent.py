
"""
被测模块：core/stream_agent.py
被测对象：stream_llm_with_tools —— 一个异步生成器（async generator），
          用"流式"的方式驱动 LLM + 工具调用循环，并逐事件产出（对应 yield 内容）：
            {"type": "token", "content": ...}      增量文本（打字机效果）
            {"type": "tool_call", "tool":..., "args":...}   请求调用工具
            {"type": "tool_result", "tool":..., "result":...} 工具执行结果
            {"type": "done", "answer": ...}         全部完成
            {"type": "error", "message": ...}       出错/超时

测试设计思路：
    - 用 MagicMock 伪造 LLM，把其 astream 方法替换成自定义的 async generator，
                    模拟"正常输出 / 请求工具 / 抛异常 / 拖时间"等行为
    - 不真实调用任何 LLM API；
    - 覆盖核心：token 累积、工具循环、未知工具兜底、超时、异常、最大轮次熔断、坏 JSON 参数兜底
"""
import asyncio
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage

from core.stream_agent import stream_llm_with_tools

def make_chunk(content="", tool_calls=None):
    """
    构造一个 mock 的 astream chunk（模拟 LLM 流式返回的 chunk 数据）。
    参数：
      content:    该片段的文本内容
      tool_calls: 工具调用片段列表，形如
                  [{"index":0, "name":"工具名", "args":"JSON字符串", "id":"调用id"}]
                  为空表示本轮纯文本、不请求工具。
    """
    chunk = MagicMock()
    chunk.content = content
    chunk.tool_call_chunks = tool_calls or []
    return chunk

async def collect(llm, tool_map=None, max_tool_loop=5, llm_timeout=5):
    """
    运行流式 agent 并收集所有事件，返回事件字典列表。
    参数：
      llm:         伪造的 LLM 对象（须有 astream 方法）
      tool_map:    工具名 → 工具对象 的映射（相当于后端 ALL_TOOLS）
      max_tool_loop: 最大工具调用轮次（防死循环）
      llm_timeout:  单轮 LLM 流式的超时秒数
    """

    events = []

    async for e in stream_llm_with_tools(
        messages=[HumanMessage(content="hi")],
        llm_getter=lambda: llm,
        tool_map=tool_map or {},
        io_executor=None,
        max_tool_loop=max_tool_loop,
        llm_timeout=llm_timeout,
    ):
        events.append(e)
    return events

class TestStreamAgent:
    """
    流式 Agent 行为测试
    通过 fake_astream 模拟"LLM 每一轮输出什么"，
    然后断言事件是否正确 。
    """

    async def test_plain_token_and_done(self):
        """LLM 无工具调用，直接输出两段文本。
        预期：按顺序推送两个 token 事件，最后以 done 结束，
        done 的 answer 等于所有 token 拼接的完整回答。"""
        llm = MagicMock()
        async def fake_astream(messages):
            yield make_chunk("你好")
            yield make_chunk("世界")

        llm.astream = fake_astream
        events = await collect(llm)

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["你好", "世界"]
        assert events[-1]["type"] == "done"
        assert events[-1]["answer"] == "你好世界"

    async def test_tool_call_loop(self):
        """第一轮 LLM 请求调用工具（defect_type_identify），
            第二轮输出最终回答。
        预期：
          - 事件序列包含 tool_call 与 tool_result
          - 工具以解析后的 dict 参数被调用一次
          - 最终 answer 是所有轮次文本的累积（"需要工具" + "最终回答"）
        """
        llm = MagicMock()
        state = {"n": 0}
        async def fake_astream(messages):
            state["n"] += 1

            if state["n"] == 1:

                yield make_chunk(
                    "需要工具",
                    tool_calls=[
                        {"index": 0,
                         "name": "defect_type_identify",
                         "args": '{"img_path": "/tmp/a.jpg"}',
                         "id": "call_1"}
                    ])
            else:

                yield make_chunk("最终回答")
        llm.astream = fake_astream

        tool = MagicMock()
        tool.invoke.return_value = "锈蚀缺陷"
        events = await collect(llm, tool_map={"defect_type_identify": tool})

        types = [e["type"] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert events[-1]["type"] == "done"

        assert events[-1]["answer"] == "需要工具最终回答"

        tool.invoke.assert_called_once_with({"img_path": "/tmp/a.jpg"})

    async def test_tool_name_not_in_map(self):
        """LLM 请求调用一个不存在的工具（no_such_tool）。
        预期：流程继续，tool_result 的结果文本包含"不存在"提示。"""
        llm = MagicMock()
        async def fake_astream(messages):
            yield make_chunk("", tool_calls=[
                {"index": 0, "name": "no_such_tool", "args": "{}", "id": "c1"}])
            yield make_chunk("回答")
        llm.astream = fake_astream

        events = await collect(llm)
        tool_results = [e for e in events if e["type"] == "tool_result"]
        assert "不存在" in tool_results[0]["result"]

    async def test_timeout_yields_error(self):
        """LLM 流式响应超过 llm_timeout。
        预期：产出 error 事件，消息含"超时"。"""
        llm = MagicMock()
        async def fake_astream(messages):
            await asyncio.sleep(0.5)
            yield make_chunk("x")
        llm.astream = fake_astream

        events = await collect(llm, llm_timeout=0.05)
        assert events[-1]["type"] == "error"
        assert "超时" in events[-1]["message"]

    async def test_exception_yields_error(self):
        """场景：LLM 流式中途抛异常。
        预期：产出 error 事件，消息包含异常原文"boom"。"""
        llm = MagicMock()
        async def fake_astream(messages):
            yield make_chunk("x")
            raise RuntimeError("boom")
        llm.astream = fake_astream

        events = await collect(llm)
        assert events[-1]["type"] == "error"
        assert "boom" in events[-1]["message"]

    async def test_max_loop_reached(self):
        """场景：LLM 每轮都坚持要调用工具，达到 max_tool_loop=2 上限。
        预期：不再继续，以 done 结束，answer 提示"最大工具调用轮次"。"""
        llm = MagicMock()
        async def fake_astream(messages):

            yield make_chunk("", tool_calls=[
                {"index": 0, "name": "defect_type_identify", "args": "{}", "id": "c"}
            ])
        llm.astream = fake_astream

        tool = MagicMock()
        tool.invoke.return_value = "结果"
        events = await collect(llm, tool_map={"defect_type_identify": tool},
                               max_tool_loop=2)
        assert events[-1]["type"] == "done"
        assert "最大工具调用轮次" in events[-1]["answer"]

    async def test_tool_args_bad_json_fallback(self):
        """LLM 返回的工具参数是非法 JSON。
        预期：不抛异常，参数解析失败后回退为空 dict传给工具，
        流程继续，最终正常 done。"""
        llm = MagicMock()
        state = {"n": 0}
        async def fake_astream(messages):
            state["n"] += 1
            if state["n"] == 1:
                yield make_chunk("", tool_calls=[
                    {"index": 0, "name": "defect_type_identify",
                     "args": "{not-json", "id": "c1"}
                ])
            else:
                yield make_chunk("ok")
        llm.astream = fake_astream

        tool = MagicMock()
        tool.invoke.return_value = "结果"
        events = await collect(llm, tool_map={"defect_type_identify": tool})
        assert events[-1]["type"] == "done"

        tool.invoke.assert_called_once_with({})
