import os
import json
import re
import logging
import hashlib
import time
import tempfile
import shutil
from typing import List, Dict, Any, Optional

import asyncio

from contextlib import asynccontextmanager

import threading

from concurrent.futures import ThreadPoolExecutor

from concurrent.futures import TimeoutError as FutureTimeoutError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
import uvicorn

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.tools import Tool
from langchain_core.documents import Document
from core.config import get_llm
from core.rag_module import get_retriever, format_docs, init_vector_store
from core.vision_module import parse_image
from core.tools_module import ALL_TOOLS
from core.reranker import Reranker
from core.query_rewriter import QueryRewriter
from core.session_manager import SessionManager
from core.rate_limiter import RateLimiter, CircuitBreaker
from core.stream_agent import stream_llm_with_tools

IO_EXECUTOR_MAX_WORKERS = 15

LLM_EXECUTOR_MAX_WORKERS = 5

CACHE_EXPIRE_MINUTES = 10

MAX_CACHE_ITEMS = 1000

LLM_TIMEOUT = 120

MAX_TOOL_LOOP = 5

ENV = os.getenv("APP_ENV", "development")
_raw_origins = os.getenv("ALLOW_CORS_ORIGINS", "*")
ALLOW_CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

retriever = None

tool_map: Dict[str, Tool] = {}

IO_EXECUTOR: ThreadPoolExecutor = None

LLM_EXECUTOR: ThreadPoolExecutor = None

TEMP_DIR = None

reranker = None
query_rewriter = None

cache_manager = None
session_manager = None

rate_limiter = RateLimiter(
    global_rate=50, global_capacity=100,
    per_ip_rate=5, per_ip_capacity=10
)
llm_circuit_breaker = CircuitBreaker(
    name="llm_api",
    failure_threshold=5,
    recovery_timeout=30
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("IndustrialMultiModalAPI")

SYSTEM_RULE_BASE = """你是一名严格遵守规则的工业质检专家。你的所有结论必须基于：
  - 工具返回的实际结果
  - 知识库中的参考文档
  - 用户提供的信息
你绝不能编造数据、猜测统计结果或给出你无法核实的建议。
工具缺失路径时主动向用户索要；工具无结果如实告知，禁止模糊推测。
"""

SYSTEM_RULE_TEXT = SYSTEM_RULE_BASE + """
【工具使用优先级与场景】
1. 批量缺陷统计：当用户要求统计某个文件夹下缺陷，调用 batch_defect_statistics，提供文件夹路径。
2. 图片分析：涉及图片识别、缺陷、OCR时使用对应图片工具。
3. 知识库问答：无工具需求时直接基于文档回答，禁止主观猜测。
"""

SYSTEM_RULE_MULTIMODAL = SYSTEM_RULE_BASE + """
可使用工具读取临时图片路径下的图片、识别缺陷、提取文字。
优先基于"图片基础描述"与知识库参考文档直接回答；仅当需要提取图片中的文字，或需确认更精确的缺陷细节时，才调用图片工具（extract_image_text / defect_type_identify）。
多图请求中，每张图片由独立流程分别分析，禁止向用户索要其他图片的路径或文件夹路径。
"""

_thread_local = threading.local()
def get_thread_llm_with_tools():


    if not hasattr(_thread_local, "llm_with_tools"):

        logger.debug(f"线程 {threading.current_thread().name} 初始化LLM实例")
        llm = get_llm()
        _thread_local.llm_with_tools = llm.bind_tools(ALL_TOOLS)

    return _thread_local.llm_with_tools

_thread_local_plain = threading.local()
def get_thread_llm_plain():


    if not hasattr(_thread_local_plain, "llm_plain"):
        logger.debug(f"线程 {threading.current_thread().name} 初始化纯LLM实例（ReAct文本协议）")
        _thread_local_plain.llm_plain = get_llm()

    return _thread_local_plain.llm_plain

def _llm_plain_invoke_worker(messages: List) -> AIMessage:


    if not llm_circuit_breaker.allow_request():
        raise Exception("LLM服务熔断中，请稍后重试")
    try:
        result = get_thread_llm_plain().invoke(messages)

        llm_circuit_breaker.record_success()
        return result
    except Exception:
        llm_circuit_breaker.record_failure()
        raise

def _batch_summary_worker(batch_result: List[dict]) -> str:


    if not llm_circuit_breaker.allow_request():
        raise Exception("LLM服务熔断中，请稍后重试")
    lines = []
    for i, r in enumerate(batch_result, 1):
        filename = r.get("filename") or f"第{i}张图片"
        answer = (r.get("answer") or "").strip()
        if not answer:
            answer = (r.get("error") or "（该图分析失败）").strip()

        lines.append(f"{i}. {filename}：{answer[:150]}")
    summary_input = "\n".join(lines)
    prompt = (
        f"以下是对 {len(batch_result)} 张图片的独立缺陷分析结论（序号与图片一一对应）：\n{summary_input}\n\n"
        "请按以下格式直接输出统计结果：\n"
        "图1：缺陷类型（严重程度）\n"
        "图2：缺陷类型（严重程度）\n"
        "……（按实际图片数量逐张列出）\n\n"
        "统计结果：\n"
        "- 缺陷类型A：N 件（图x、图y）\n"
        "- 缺陷类型B：N 件（图z）\n"
        "要求：每个缺陷类型的数量必须与其后列出的图号个数完全一致；"
        "若某张图含多种缺陷，请分别列出并计数；"
        "若全部合格请直接说明。"
    )
    try:
        result = get_thread_llm_plain().invoke([HumanMessage(content=prompt)])
        llm_circuit_breaker.record_success()
        return result.content if hasattr(result, "content") else str(result)
    except Exception:
        llm_circuit_breaker.record_failure()
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):


    global retriever, tool_map, IO_EXECUTOR, LLM_EXECUTOR, TEMP_DIR, query_rewriter, reranker, cache_manager, session_manager

    logger.info("应用启动，开始初始化全局资源...")
    try:

        IO_EXECUTOR = ThreadPoolExecutor(
            max_workers=IO_EXECUTOR_MAX_WORKERS,
            thread_name_prefix="io_worker")
        LLM_EXECUTOR = ThreadPoolExecutor(
            max_workers=LLM_EXECUTOR_MAX_WORKERS,
            thread_name_prefix="llm_worker")
        logger.info(f"线程池初始化完成：IO池 {IO_EXECUTOR_MAX_WORKERS}，LLM池 {LLM_EXECUTOR_MAX_WORKERS}")

        from core.cache_manager import CacheManager
        redis_url = os.getenv("REDIS_URL")
        cache_manager = CacheManager(redis_url=redis_url, fallback_to_local=True)
        logger.info(f"缓存管理器初始化完成，后端: {cache_manager.backend_type}")  

        tool_map = {tool.name: tool for tool in ALL_TOOLS}

        test_llm = get_llm()
        test_llm.bind_tools(ALL_TOOLS)
        logger.info("向量库、LLM配置、工具资源初始化完成")

        retriever = get_retriever(k=10)
        reranker = Reranker(top_n=3, use_api=True)
        query_rewriter = QueryRewriter(get_llm())
        session_manager = SessionManager()

        TEMP_DIR = tempfile.mkdtemp(prefix="industrial_mm_")
        logger.info(f"专用临时目录已创建: {TEMP_DIR}")
    except Exception as init_err:
        logger.critical("全局资源初始化失败，服务无法启动", exc_info=True)
        raise RuntimeError("核心资源初始化异常") from init_err

    yield

    logger.info("应用关闭，开始清理资源...")

    IO_EXECUTOR.shutdown(wait=True)
    LLM_EXECUTOR.shutdown(wait=True)
    logger.info("所有线程池已关闭")

    if TEMP_DIR and os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)

            logger.info(f"临时目录已删除: {TEMP_DIR}")
        except Exception as e:
            logger.warning(f"删除临时目录失败: {e}")

    logger.info("资源清理完成，应用退出")

app = FastAPI(
    title="多模态问答系统API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")

async def global_exception_handler(request: Request, call_next):


    start_time = time.time()
    try:
        response = await call_next(request)

        cost = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Request {request.method} {request.url.path} | cost {cost}ms")
        return response
    except Exception as e:

        if isinstance(e, HTTPException):
            raise

        cost = round((time.time() - start_time) * 1000, 2)
        logger.error( f"Request {request.method} {request.url.path} failed, cost {cost}ms, error: {str(e)}",
            exc_info=True )
        return JSONResponse(status_code=500,
            content={"code": 500, "msg": f"服务内部异常: {str(e)}", "data": None} )

@app.exception_handler(HTTPException)

async def http_exception_handler(request: Request, exc: HTTPException):


    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "msg": exc.detail, "data": None}

    )

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"

    allowed, reason = rate_limiter.check(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"code": 429, "msg": reason, "data": None}
        )
    return await call_next(request)

def _normalize_query(query: str) -> str:

    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    return query

def _make_rag_cache_key(query: str) -> str:

    norm_query = _normalize_query(query)
    query_hash = hashlib.md5(norm_query.encode("utf-8")).hexdigest()
    return f"rag:{query_hash}"

def _make_image_cache_key(file_md5: str, prompt: str) -> str:

    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return f"img:{file_md5}:{prompt_hash}"

def retrieve_with_pipeline(query, do_rewrite=True):

    queries = query_rewriter.rewrite(query, decompose=False) if do_rewrite else [query]
    all_docs, seen = [], set()
    for q in queries:
        for d in retriever.invoke(q):
            if d.page_content not in seen:
                seen.add(d.page_content)
                all_docs.append(d)
    return reranker.rerank(query, all_docs, top_n=3)

def calc_file_md5(file_bytes: bytes) -> str:


    return hashlib.md5(file_bytes).hexdigest()


REACT_FORMAT_INSTRUCTION = """你必须严格遵循 ReAct 格式逐轮输出，每一轮至多执行一个动作。

可用工具（Action 的取值只能是下列工具名之一）：
{tool_descriptions}

格式模板：
Thought: <你的推理：当前掌握了什么、还缺什么、下一步为什么这样做>
Action: <工具名；若已无需调用工具，则不要输出 Action，直接进入 Final Answer>
Action Input: <该工具入参，必须是合法 JSON，例如 {{"img_path": "F:/data/a.jpg"}}>

系统会执行 Action 并返回 Observation（Observation 由系统生成，严禁你自行编造）。
随后你继续 Thought，可如此重复多轮；当信息充分时输出：
Thought: <总结推理>
Final Answer: <面向用户的最终回答>

硬性要求：
1. Action Input 必须是合法 JSON，不要加 markdown 代码块；每轮只能有一个 Action。
2. 结论只能来自 Observation 与已给上下文，禁止编造数据；工具缺少路径等必要参数时不要反复尝试，直接在 Final Answer 中向用户索要。
"""

def _build_tool_descriptions() -> str:

    desc_lines = []
    for tool_obj in ALL_TOOLS:
        first_line = ""
        if tool_obj.description:
            first_line = tool_obj.description.strip().splitlines()[0]
        desc_lines.append(
            f"- {tool_obj.name}: {first_line}；入参Schema: {json.dumps(tool_obj.args, ensure_ascii=False)}"
        )
    return "\n".join(desc_lines)

def _parse_react_output(text: str) -> dict:


    result = {"thought": "", "action": None, "action_input": {}, "raw_input": "", "final_answer": None}
    raw = (text or "").strip()
    if not raw:
        return result

    final_match = re.search(r"(?:final\s*answer|最终答案)\s*[:：]\s*(.*)$",
                            raw, re.IGNORECASE | re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()

        result["thought"] = raw[:final_match.start()].strip()
        return result

    thought_match = re.search(
        r"thought\s*[:：]\s*(.*?)(?=action\s*input\s*[:：]|action\s*[:：]|final\s*answer\s*[:：]|最终答案\s*[:：]|$)",
        raw, re.IGNORECASE | re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    for m in re.finditer(r"action\s*[:：]\s*([^\n\r]*)", raw, re.IGNORECASE):
        candidate = m.group(1).strip().strip("`\"' ")
        if candidate and not candidate.lower().startswith("input"):

            result["action"] = candidate
            break

    input_match = re.search(
        r"action\s*input\s*[:：]\s*(.*?)(?=thought\s*[:：]|observation\s*[:：]|action\s*[:：]|final\s*answer\s*[:：]|最终答案\s*[:：]|$)",
        raw, re.IGNORECASE | re.DOTALL)
    if input_match:
        input_raw = input_match.group(1).strip().strip("`").strip()
        if input_raw.lower().startswith("json"):
            input_raw = input_raw[4:].strip().strip("`").strip()
        result["raw_input"] = input_raw
        try:
            result["action_input"] = json.loads(input_raw)
        except Exception:
            result["action_input"] = {}
    return result

def _coerce_single_str_arg(tool_name: str, raw_text: str) -> dict:

    tool_obj = tool_map.get(tool_name)
    raw_text = (raw_text or "").strip().strip("`\"' ")
    if tool_obj is not None and raw_text:
        try:
            props = tool_obj.args.get("properties", {})

            str_props = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"]

            if len(str_props) == 1:

                return {str_props[0]: raw_text}
        except Exception:
            pass
    return {}

def run_react_agent(input_messages: List[BaseMessage]) -> tuple[str, List[dict]]:


    messages = input_messages.copy()

    react_instruction = REACT_FORMAT_INSTRUCTION.format(tool_descriptions=_build_tool_descriptions())

    injected = False
    for idx, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            messages[idx] = SystemMessage(content=f"{msg.content}\n\n{react_instruction}")

            injected = True
            break
    if not injected:
        messages.insert(0, SystemMessage(content=react_instruction))

    tool_steps: List[dict] = []
    loop_count = 0
    last_action_sig = None
    reflection_count = 0
    MAX_REFLECTIONS = 2

    logger.info(f"启动ReAct推理循环，最大轮次 {MAX_TOOL_LOOP}")
    while loop_count < MAX_TOOL_LOOP:
        loop_count += 1
        logger.info(f"===== ReAct 第 {loop_count} 轮 =====")

        try:
            future = LLM_EXECUTOR.submit(_llm_plain_invoke_worker, messages)
            ai_msg: AIMessage = future.result(timeout=LLM_TIMEOUT)
        except FutureTimeoutError:
            logger.error(f"第{loop_count}轮LLM调用超时，超过{LLM_TIMEOUT}秒")
            return "大模型响应超时，请稍后重试", tool_steps
        except Exception as llm_err:
            logger.error(f"第{loop_count}轮LLM调用失败", exc_info=True)
            return f"大模型调用异常：{str(llm_err)}", tool_steps

        if isinstance(ai_msg.content, str):
            text = ai_msg.content.strip()
        else:
            text = str(ai_msg.content).strip()
        messages.append(AIMessage(content=text))
        parsed = _parse_react_output(text)

        if parsed["thought"]:
            tool_steps.append({"round": loop_count, "step": "thought",
                               "content": parsed["thought"][:500]})

        if parsed["final_answer"] is not None:
            logger.info("ReAct 输出 Final Answer，结束循环")
            return parsed["final_answer"], tool_steps

        action = parsed["action"]

        if not action:
            logger.warning("ReAct输出未包含Action/Final Answer")
            return text or "模型未返回有效回答", tool_steps

        tool_args = parsed["action_input"]
        if not tool_args and parsed["raw_input"]:
            tool_args = _coerce_single_str_arg(action, parsed["raw_input"])


        sig = (action, json.dumps(tool_args, ensure_ascii=False, sort_keys=True))

        if last_action_sig is not None and sig == last_action_sig:

            reflection_count += 1
            logger.warning(f"检测到无进展循环，连续第{reflection_count}次")

            if reflection_count >= MAX_REFLECTIONS:
                logger.warning("连续无进展次数超过阈值，强制收尾")
                messages.append(SystemMessage(
                    content="【系统提示】多次尝试未获得新信息，请基于已有信息直接输出 Final Answer，不要再调用工具。"))
                try:
                    fut = LLM_EXECUTOR.submit(_llm_plain_invoke_worker, messages)
                    final_msg = fut.result(timeout=LLM_TIMEOUT)
                    fparsed = _parse_react_output(final_msg.content or "")
                    answer = fparsed["final_answer"] or (final_msg.content or "").strip()
                    return answer or "多次尝试后仍无法获取足够信息，请补充资料后重试", tool_steps
                except Exception as force_err:
                    logger.error(f"强制收尾调用失败: {force_err}")
                    return "多次尝试后仍无法获取足够信息，请补充资料后重试", tool_steps

            messages.append(SystemMessage(
                content="【系统提示】你上一轮以完全相同的入参调用了同一工具，不会带来新信息。"
                        "请反思：是否应更换工具、修正入参，或基于已有信息直接输出 Final Answer？"))
            continue
        reflection_count = 0
        last_action_sig = sig

        tool_steps.append({"round": loop_count, "step": "action", "tool": action, "args": tool_args})
        logger.info(f"ReAct Action: {action}，入参: {str(tool_args)[:200]}")
        if action not in tool_map:
            observation = f"错误：不存在名为 {action} 的工具，可用工具：{list(tool_map.keys())}"
            logger.warning(observation)
        else:
            try:
                observation = tool_map[action].invoke(tool_args)
                logger.info(f"工具 {action} 执行成功，结果长度 {len(str(observation))}")
            except Exception as tool_e:
                observation = f"工具执行异常：{str(tool_e)}"
                logger.error(f"工具 {action} 执行报错", exc_info=True)

        tool_steps.append({"round": loop_count, "step": "observation",
                           "tool": action, "result": str(observation)[:500]})

        messages.append(HumanMessage(content=f"Observation: {observation}"))

    logger.warning(f"ReAct 达到最大轮次 {MAX_TOOL_LOOP}")
    return f"【提示：已达到最大推理轮次({MAX_TOOL_LOOP}轮)，回答可能不完整】", tool_steps


STALE_TMP_PATH_PATTERN = re.compile(r"(?:图片本地临时路径|文件夹)：[^\s,;）)]+")
def _sanitize_history(messages) -> List[BaseMessage]:


    if not isinstance(messages, (list, tuple)):
        return []
    cleaned = []
    for msg in messages:
        if isinstance(msg.content, str) and STALE_TMP_PATH_PATTERN.search(msg.content):
            msg = msg.__class__(content=STALE_TMP_PATH_PATTERN.sub(
                "（历史图片已过期，不可再调用工具读取）", msg.content))
        cleaned.append(msg)
    return cleaned

def _persist_batch_to_session(session, batch_result: List[dict], batch_summary: str) -> bool:


    ordered = sorted(
        [r for r in batch_result if isinstance(r, dict)],
        key=lambda r: r.get("img_index", 0)
    )
    for r in ordered:
        if r.get("answer"):
            session.add_message(AIMessage(
                content=f"【图片 {r.get('filename', '')}】{r['answer']}"))
    if batch_summary:
        session.add_message(AIMessage(content=f"【批量缺陷统计】{batch_summary}"))
    return session.should_summarize()


def _chat_sync_handler(session, question):


    rag_docs = retrieve_with_pipeline(question)
    context = format_docs(rag_docs)

    messages = session.get_context_messages(SYSTEM_RULE_TEXT)

    messages.insert(-1, SystemMessage(content=f"知识库参考文档：\n{context}"))

    answer, tool_steps = run_react_agent(messages)

    session.add_message(AIMessage(content=answer))

    return {
        "answer": answer,
        "tool_steps": tool_steps,
        "need_compress": session.should_summarize(),
        "docs": [d.page_content for d in rag_docs]
    }

def process_single_image_sync(img_bytes: bytes, filename: str, question: str,
                              img_index: int = 0, total_images: int = 1,
                              history_messages: Optional[List[BaseMessage]] = None) -> dict:


    tmp_path = ""
    try:

        suffix = os.path.splitext(filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        logger.info(f"临时图片保存路径: {tmp_path}")

        file_md5 = calc_file_md5(img_bytes)
        parse_prompt = "详细描述图片中零件的外观、缺陷特征"

        img_cache_key = _make_image_cache_key(file_md5, parse_prompt)

        def load_image_desc():

            logger.info("未命中图片缓存，执行视觉解析")
            return parse_image(tmp_path, parse_prompt)

        img_desc = cache_manager.get_or_set(
            key=img_cache_key,
            loader_func=load_image_desc,
            ttl=CACHE_EXPIRE_MINUTES * 60,
            use_lock=True
        )
        logger.info(f"图片md5 {file_md5} 处理完成")

        search_query = f"{img_desc}\n{question}"
        rag_cache_key = _make_rag_cache_key(search_query)

        def load_rag_docs():

            docs = retrieve_with_pipeline(search_query)

            return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

        docs_content = cache_manager.get_or_set(
            key=rag_cache_key,
            loader_func=load_rag_docs,
            ttl=CACHE_EXPIRE_MINUTES * 60,
            use_lock=True
        )


        docs = [Document(page_content=item["content"], metadata=item.get("metadata", {}))
                    for item in docs_content]
        context = format_docs(docs)

        batch_hint = ""
        if total_images > 1:
            batch_hint = (
                f"本次请求共 {total_images} 张图片，你正在处理第 {img_index + 1}/{total_images} 张（文件：{filename}）。"
                f"其他图片已由系统自动并行处理，你无需读取、索要或分析其他图片。"
                f"即使问题中提到「这些图片」「所有图片」等复数表述，也请只完成当前这一张图片的缺陷分析，"
                f"回答结尾绝对不要索要任何文件路径或文件夹。\n"
            )


        messages = list(history_messages) if history_messages else [SystemMessage(content=SYSTEM_RULE_MULTIMODAL)]
        messages.append(HumanMessage(content=f"""知识库参考文档：{context}
{batch_hint}图片本地临时路径：{tmp_path}
图片基础描述：{img_desc}
用户问题：{question}
"""))
        answer, tool_steps = run_react_agent(messages)

        return {
            "filename": filename,
            "img_md5": file_md5,
            "image_description": img_desc,
            "answer": answer,
            "docs": [d.page_content for d in docs],
            "tool_steps": tool_steps
        }
    except Exception as e:
        logger.error(f"单图处理失败: {filename}", exc_info=True)
        return {"filename": filename, "error": str(e), "answer": None, "tool_steps": []}

    finally:

        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"已清理临时图片: {tmp_path}")
            except Exception as clean_e:
                logger.warning(f"临时文件删除失败 {tmp_path}: {str(clean_e)}")


@app.post("/api/chat")
async def chat(
    question: str = Form(...),
    session_id: str = Form(...),
    stream: bool = Form(False),
    images: List[UploadFile] = File(None)
):


    if images:
        logger.info(f"接收多模态请求，图片数量: {len(images)}, 问题: {question[:100]}...")
        session = session_manager.get_or_create(session_id)
        sem = asyncio.Semaphore(4)


        img_names = ", ".join(img.filename or f"image_{i}.jpg" for i, img in enumerate(images))
        session.add_message(HumanMessage(
            content=f"{question}\n（本次请求上传 {len(images)} 张图片：{img_names}）"))
        history_messages = _sanitize_history(
            session.get_context_messages(SYSTEM_RULE_MULTIMODAL))

        async def process_one(idx: int, upload_img: UploadFile) -> dict:

            async with sem:
                img_bytes = await upload_img.read()
                filename = upload_img.filename or f"image_{idx}.jpg"
                loop = asyncio.get_running_loop()

                result = await loop.run_in_executor(
                    IO_EXECUTOR,
                    process_single_image_sync,
                    img_bytes,
                    filename,
                    question,
                    idx,
                    len(images),
                    history_messages
                )
                result["img_index"] = idx
                return result

        if not stream:
            try:
                tasks = [process_one(idx, img) for idx, img in enumerate(images)]

                batch_result = await asyncio.gather(*tasks)

                batch_summary = ""
                if len(images) > 1:
                    try:
                        loop = asyncio.get_running_loop()
                        batch_summary = await loop.run_in_executor(
                            LLM_EXECUTOR, _batch_summary_worker, batch_result
                        )
                    except Exception as e:
                        logger.error("批量缺陷统计汇总失败，降级为空", exc_info=True)
                        batch_summary = ""

                loop = asyncio.get_running_loop()
                need_compress = _persist_batch_to_session(session, batch_result, batch_summary)
                if need_compress:
                    loop.run_in_executor(IO_EXECUTOR, session.compress_history, get_llm())

                logger.info(f"多模态{len(images)}张图片全部处理完成")
                return {
                    "code": 200,
                    "session_id": session_id,
                    "mode": "multimodal",
                    "batch_size": len(images),
                    "question": question,
                    "multimodalResults": batch_result,
                    "batchSummary": batch_summary
                }
            except Exception as e:
                logger.error("多模态推理接口处理异常", exc_info=True)
                raise HTTPException(status_code=500, detail=f"多模态推理失败: {str(e)}")

        async def multimodal_event_generator():

            tasks = [asyncio.create_task(process_one(idx, img)) for idx, img in enumerate(images)]
            pending = set(tasks)
            done_results = []
            try:
                yield f"data: {json.dumps({'type': 'start', 'session_id': session_id, 'mode': 'multimodal'}, ensure_ascii=False)}\n\n"
                while pending:
                    finished, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED)
                    for t in finished:
                        try:
                            r = t.result()
                        except Exception as task_e:
                            logger.error(f"单图处理任务异常: {task_e}", exc_info=True)
                            r = {"filename": "unknown", "error": str(task_e), "answer": None}
                        done_results.append(r)
                        yield f"data: {json.dumps({'type': 'image_done', **r}, ensure_ascii=False)}\n\n"

                batch_summary = ""
                if len(images) > 1:
                    try:
                        loop = asyncio.get_running_loop()
                        ordered_results = sorted(done_results, key=lambda r: r.get("img_index", 0))
                        batch_summary = await loop.run_in_executor(
                            LLM_EXECUTOR, _batch_summary_worker, ordered_results)
                    except Exception as e:
                        logger.error("批量缺陷统计汇总失败，降级为空", exc_info=True)
                        batch_summary = ""
                    yield f"data: {json.dumps({'type': 'batch_summary', 'batchSummary': batch_summary}, ensure_ascii=False)}\n\n"

                loop = asyncio.get_running_loop()
                need_compress = _persist_batch_to_session(session, done_results, batch_summary)
                if need_compress:
                    loop.run_in_executor(IO_EXECUTOR, session.compress_history, get_llm())


                final_answer = batch_summary
                if not final_answer:
                    for r in sorted(done_results, key=lambda x: x.get("img_index", 0)):
                        if r.get("answer"):
                            final_answer = r["answer"]
                            break
                yield f"data: {json.dumps({'type': 'done', 'mode': 'multimodal', 'batch_size': len(images), 'session_id': session_id, 'answer': final_answer or ''}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"多模态流式输出异常: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            multimodal_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    session = session_manager.get_or_create(session_id)
    session.add_message(HumanMessage(content=question))

    if not stream:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            IO_EXECUTOR, _chat_sync_handler, session, question
        )

        if result["need_compress"]:
            loop.run_in_executor(IO_EXECUTOR, session.compress_history, get_llm())
        return {
            "code": 200,
            "answer": result["answer"],
            "session_id": session_id,
            "tool_steps": result["tool_steps"],
            "docs": result["docs"]
        }

    messages = session.get_context_messages(SYSTEM_RULE_TEXT)


    loop = asyncio.get_running_loop()
    rag_docs = await loop.run_in_executor(IO_EXECUTOR, retrieve_with_pipeline, question)
    context = format_docs(rag_docs)
    messages.insert(-1, SystemMessage(content=f"知识库参考文档：\n{context}"))

    async def event_generator():

        full_answer = ""
        try:
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id}, ensure_ascii=False)}\n\n"
            async for event in stream_llm_with_tools(
                messages=messages,
                llm_getter=get_thread_llm_with_tools,
                tool_map=tool_map,
                io_executor=IO_EXECUTOR,
                max_tool_loop=MAX_TOOL_LOOP,
                llm_timeout=LLM_TIMEOUT
            ):
                if event["type"] == "token":
                    full_answer += event["content"]
                elif event["type"] == "done":
                    full_answer = event["answer"]

                    session.add_message(AIMessage(content=full_answer))

                    if session.should_summarize():
                        loop = asyncio.get_running_loop()
                        loop.run_in_executor(
                            IO_EXECUTOR,
                            session.compress_history,
                            get_llm()
                        )

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"流式输出异常: {e}", exc_info=True)
            err_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/health")
async def health_check():

    return {
        "code": 200,
        "status": "healthy",
        "timestamp": time.time()
    }

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):


    session_manager.delete(session_id)
    logger.info(f"会话已删除: {session_id}")
    return {"code": 200, "message": "session deleted", "session_id": session_id}

if __name__ == "__main__":

    reload_enabled = ENV == "development" 

    logger.info(f"启动多模态问答API服务，环境: {ENV}，地址 127.0.0.1:8000，reload: {reload_enabled}")

    uvicorn.run(

        "api_llm_pro_latest:app",


        host="127.0.0.1",

        port=8000,

        reload=reload_enabled
    )
