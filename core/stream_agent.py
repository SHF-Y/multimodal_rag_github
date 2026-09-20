
import json
import asyncio
import logging
from typing import List, AsyncGenerator, Dict, Any
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

logger = logging.getLogger("StreamAgent")

async def stream_llm_with_tools(
    messages: List[BaseMessage],
    llm_getter,
    tool_map: Dict[str, Any],
    io_executor, 
    max_tool_loop: int = 5,
    llm_timeout: int = 120
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    流式 LLM 调用 + 工具调用循环

    Args:
        messages: 初始消息列表（SystemMessage + HumanMessage + 历史）
        llm_getter: 返回已绑定工具的LLM实例的函数（线程安全）
        tool_map: 工具名 -> Tool实例 的映射
        io_executor: 同步工具执行用的线程池
        max_tool_loop: 最大工具调用轮次
        llm_timeout: 单轮LLM流式调用超时（秒）

    Yields:
        事件字典，type 取值：
        - "token":       增量文本 {"type": "token", "content": "..."}
        - "tool_call":   即将调用工具 {"type": "tool_call", "tool": "...", "args": {...}}
        - "tool_result": 工具执行结果 {"type": "tool_result", "tool": "...", "result": "..."}
        - "done":        全部完成 {"type": "done", "answer": "完整回答"}
        - "error":       出错 {"type": "error", "message": "..."}
    """

    llm = llm_getter()
    full_answer = ""
    loop_count = 0

    while loop_count < max_tool_loop:
        loop_count += 1
        round_content = ""
        logger.info(f"流式Agent第 {loop_count} 轮")

        accumulated_tc: Dict[int, Dict[str, str]] = {}

        try:
            async with asyncio.timeout(llm_timeout):

                async for chunk in llm.astream(messages):

                    if chunk.content:
                        full_answer += chunk.content
                        yield {"type": "token", "content": chunk.content}

                    if chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:

                            idx = tc.get("index", 0)

                            if idx not in accumulated_tc:
                                accumulated_tc[idx] = {"name": "", "args_str": "", "id": ""}

                            if tc.get("name"):
                                accumulated_tc[idx]["name"] = tc["name"]
                            if tc.get("id"):
                                accumulated_tc[idx]["id"] = tc["id"]

                            if tc.get("args"):
                                accumulated_tc[idx]["args_str"] += tc["args"]

        except asyncio.TimeoutError:

            yield {"type": "error", "message": f"LLM响应超时（{llm_timeout}秒）"}
            return
        except Exception as e:
            logger.error(f"流式LLM调用失败: {e}", exc_info=True)
            yield {"type": "error", "message": f"LLM调用异常: {str(e)}"}
            return

        parsed_tool_calls = []
        for idx, tc_data in accumulated_tc.items():
            if not tc_data["name"]:
                continue
            try:
                args = json.loads(tc_data["args_str"]) if tc_data["args_str"] else {}
            except json.JSONDecodeError:
                logger.warning(f"工具参数JSON解析失败: {tc_data['args_str']}")
                args = {}
            parsed_tool_calls.append({
                "name": tc_data["name"],
                "args": args,
                "id": tc_data["id"] or f"call_{idx}_{loop_count}"
            })

        ai_message = AIMessage(
            content=round_content,
            tool_calls=[ {"name": tc["name"], "args": tc["args"], "id": tc["id"]} for tc in parsed_tool_calls]
        )
        messages.append(ai_message)

        if not parsed_tool_calls:
            logger.info("流式Agent：无工具调用，结束")
            yield {"type": "done", "answer": full_answer}
            return

        for tc in parsed_tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            call_id = tc["id"]

            yield {"type": "tool_call", "tool": tool_name, "args": tool_args}

            try:
                if tool_name not in tool_map:
                    tool_result = f"错误：不存在名为 {tool_name} 的工具"
                else:
                    loop = asyncio.get_running_loop()

                    tool_result = await loop.run_in_executor(
                        io_executor,
                        tool_map[tool_name].invoke,
                        tool_args
                    )
            except Exception as e:
                tool_result = f"工具执行异常：{str(e)}"
                logger.error(f"工具 {tool_name} 执行失败: {e}", exc_info=True)

            yield {"type": "tool_result","tool": tool_name,"result": str(tool_result)[:500] }

            tool_msg = ToolMessage(
                content=str(tool_result),
                tool_call_id=call_id,
                name=tool_name
            )
            messages.append(tool_msg)

    logger.warning(f"流式Agent达到最大轮次 {max_tool_loop}")
    yield {
        "type": "done",
        "answer": f"【提示：已达到最大工具调用轮次({max_tool_loop}轮)，回答可能不完整】\n{full_answer}"
    }