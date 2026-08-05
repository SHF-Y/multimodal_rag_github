import os
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
import uvicorn

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.tools import Tool

from core.config import get_llm
from core.rag_module import get_retriever, format_docs, init_vector_store
from core.vision_module import parse_image
from core.tools_module import ALL_TOOLS

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

CACHE_LOCK = threading.Lock()
IMAGE_CACHE: Dict[str, Dict[str, Any]] = {}
RAG_CACHE: Dict[str, Dict[str, Any]] = {}

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
"""

_thread_local = threading.local()
def get_thread_llm_with_tools():
    if not hasattr(_thread_local, "llm_with_tools"):
        logger.debug(f"线程 {threading.current_thread().name} 初始化独立LLM实例")
        llm = get_llm()
        _thread_local.llm_with_tools = llm.bind_tools(ALL_TOOLS)
    return _thread_local.llm_with_tools

def _llm_invoke_worker(messages: List) -> AIMessage:
    llm = get_thread_llm_with_tools()
    return llm.invoke(messages)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, tool_map, IO_EXECUTOR, LLM_EXECUTOR, TEMP_DIR
    logger.info("应用启动，开始初始化全局资源...")
    try:
        IO_EXECUTOR = ThreadPoolExecutor(
            max_workers=IO_EXECUTOR_MAX_WORKERS,
            thread_name_prefix="io_worker")
        LLM_EXECUTOR = ThreadPoolExecutor(
            max_workers=LLM_EXECUTOR_MAX_WORKERS,
            thread_name_prefix="llm_worker")
        logger.info(f"线程池初始化完成：IO池 {IO_EXECUTOR_MAX_WORKERS}，LLM池 {LLM_EXECUTOR_MAX_WORKERS}")
        init_vector_store()
        retriever = get_retriever()
        tool_map = {tool.name: tool for tool in ALL_TOOLS}
        test_llm = get_llm()
        test_llm.bind_tools(ALL_TOOLS)
        logger.info("向量库、LLM配置、工具资源初始化完成")
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

def _normalize_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    return query

def _enforce_cache_limit(cache_dict: Dict, max_items: int):
    if len(cache_dict) <= max_items:
        return
    sorted_items = sorted(cache_dict.items(), key=lambda x: x[1]["expire"])
    delete_count = int(max_items * 0.2) + 1
    for k, _ in sorted_items[:delete_count]:
        del cache_dict[k]
    logger.warning(f"缓存超出容量上限，淘汰最旧 {delete_count} 条，当前剩余 {len(cache_dict)} 条")

def get_cached_image_desc(file_md5: str, prompt: str) -> Optional[str]:
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    cache_key = f"{file_md5}:{prompt_hash}"
    now = time.time()
    with CACHE_LOCK:
        item = IMAGE_CACHE.get(cache_key)
        if not item:
            return None
        if item["expire"] < now:
            del IMAGE_CACHE[cache_key]
            return None
        return item["desc"]

def set_image_cache(file_md5: str, prompt: str, desc: str):
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    cache_key = f"{file_md5}:{prompt_hash}"
    expire = time.time() + CACHE_EXPIRE_MINUTES * 60
    with CACHE_LOCK:
        IMAGE_CACHE[cache_key] = {"desc": desc, "expire": expire}
        _enforce_cache_limit(IMAGE_CACHE, MAX_CACHE_ITEMS)

def get_cached_rag_docs(query: str) -> Optional[List[Any]]:
    norm_query = _normalize_query(query)
    cache_key = hashlib.md5(norm_query.encode("utf-8")).hexdigest()
    now = time.time()
    with CACHE_LOCK:
        item = RAG_CACHE.get(cache_key)
        if not item:
            return None
        if item["expire"] < now:
            del RAG_CACHE[cache_key]
            return None
        return item["docs"]

def set_rag_cache(query: str, docs: List[Any]):
    norm_query = _normalize_query(query)
    cache_key = hashlib.md5(norm_query.encode("utf-8")).hexdigest()
    expire = time.time() + CACHE_EXPIRE_MINUTES * 60
    with CACHE_LOCK:
        RAG_CACHE[cache_key] = {"docs": docs, "expire": expire}
        _enforce_cache_limit(RAG_CACHE, MAX_CACHE_ITEMS)

def calc_file_md5(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

def run_llm_with_manual_tools(input_messages: List[BaseMessage]) -> tuple[str, List[dict]]:
    messages = input_messages.copy()
    tool_steps = []
    loop_count = 0
    logger.info(f"启动LLM工具循环，最大轮次 {MAX_TOOL_LOOP}")

    while loop_count < MAX_TOOL_LOOP:
        loop_count += 1
        logger.info(f"===== 工具循环第 {loop_count} 轮 =====")
        try:
            future = LLM_EXECUTOR.submit(_llm_invoke_worker, messages)
            ai_msg: AIMessage = future.result(timeout=LLM_TIMEOUT)
            messages.append(ai_msg)
            content_preview = ai_msg.content[:200] if ai_msg.content else "[无文本内容，仅工具调用]"
            logger.info(f"LLM返回内容片段: {content_preview}")
        except FutureTimeoutError:
            logger.error(f"第{loop_count}轮LLM调用超时，超过{LLM_TIMEOUT}秒")
            return "大模型响应超时，请稍后重试", tool_steps
        except Exception as llm_err:
            logger.error(f"第{loop_count}轮LLM调用失败", exc_info=True)
            return f"大模型调用异常：{str(llm_err)}", tool_steps

        if not ai_msg.tool_calls:
            logger.info("LLM无工具调用，结束循环")
            break

        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            call_id = tool_call["id"]
            logger.info(f"准备调用工具 {tool_name}, 参数: {str(tool_args)[:200]}")
            tool_steps.append({
                "round": loop_count, "step": "tool_call",
                "tool": tool_name, "args": tool_args, "id": call_id
            })

            try:
                if tool_name not in tool_map:
                    tool_result = f"错误：不存在名为 {tool_name} 的工具"
                    logger.warning(tool_result)
                else:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                    logger.info(f"工具 {tool_name} 执行成功，结果长度: {len(str(tool_result))}")
            except Exception as tool_e:
                tool_result = f"工具执行异常：{str(tool_e)}"
                logger.error(f"工具 {tool_name} 执行报错", exc_info=True)

            tool_steps.append({
                "round": loop_count,
                "step": "tool_result",
                "tool": tool_name,
                "result": str(tool_result)[:500],
                "tool_call_id": call_id
            })

            tool_msg = ToolMessage(content=str(tool_result), tool_call_id=call_id, name=tool_name)
            messages.append(tool_msg)

    final_answer = "模型未返回有效回答"
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and str(msg.content).strip():
            final_answer = msg.content
            break

    if loop_count >= MAX_TOOL_LOOP:
        final_answer = f"【提示：已达到最大工具调用轮次({MAX_TOOL_LOOP}轮)，回答可能不完整】\n{final_answer}"
        logger.warning(f"工具循环达到最大轮次{MAX_TOOL_LOOP}")

    logger.info(f"工具循环结束，生成最终回答长度: {len(final_answer)}")
    return final_answer, tool_steps

def _text_rag_sync_handler(question: str) -> dict:
    docs = get_cached_rag_docs(question)
    if docs is None:
        logger.info("未命中RAG缓存，执行向量库检索")
        docs = retriever.invoke(question)
        set_rag_cache(question, docs)
    context = format_docs(docs)

    messages = [ SystemMessage(content=SYSTEM_RULE_TEXT),
                 HumanMessage(content=f"知识库参考文档：\n{context}\n\n用户问题：{question}")]
    answer, tool_steps = run_llm_with_manual_tools(messages)

    return {
        "code": 200,
        "answer": answer,
        "docs": [d.page_content for d in docs],
        "agent_steps": tool_steps
    }

@app.post("/api/text_rag")
async def text_rag_query(question: str = Form(...)):
    logger.info(f"接收文本RAG请求，问题: {question[:100]}...")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            IO_EXECUTOR,
            _text_rag_sync_handler,
            question
        )
        return result
    except Exception as e:
        logger.error("text_rag接口业务异常", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文本问答处理失败: {str(e)}")

def process_single_image_sync(img_bytes: bytes, filename: str, question: str) -> dict:
    tmp_path = ""
    try:
        suffix = os.path.splitext(filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        logger.info(f"临时图片保存路径: {tmp_path}")

        file_md5 = calc_file_md5(img_bytes)
        parse_prompt = "详细描述图片中零件的外观、缺陷特征"

        img_desc = get_cached_image_desc(file_md5, parse_prompt)
        if img_desc is None:
            logger.info("未命中图片缓存，执行视觉解析")
            img_desc = parse_image(tmp_path, parse_prompt)
            set_image_cache(file_md5, parse_prompt, img_desc)
        else:
            logger.info(f"图片md5 {file_md5} 命中缓存")

        search_query = f"{img_desc}\n{question}"
        docs = get_cached_rag_docs(search_query)
        if docs is None:
            docs = retriever.invoke(search_query)
            set_rag_cache(search_query, docs)
        context = format_docs(docs)

        messages = [SystemMessage(content=SYSTEM_RULE_MULTIMODAL),
                    HumanMessage(content=f"""知识库参考文档：{context}
图片本地临时路径：{tmp_path}
图片基础描述：{img_desc}
用户问题：{question}
""")
        ]
        answer, tool_steps = run_llm_with_manual_tools(messages)

        return {
            "filename": filename,
            "img_md5": file_md5,
            "image_description": img_desc,
            "answer": answer,
            "docs": [d.page_content for d in docs],
            "agent_steps": tool_steps
        }
    except Exception as e:
        logger.error(f"单图处理失败: {filename}", exc_info=True)
        return {"filename": filename, "error": str(e), "answer": None, "agent_steps": []}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"已清理临时图片: {tmp_path}")
            except Exception as clean_e:
                logger.warning(f"临时文件删除失败 {tmp_path}: {str(clean_e)}")

@app.post("/api/multimodal_rag")
async def multimodal_rag_query(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    filename = image.filename or "upload.jpg"
    logger.info(f"接收单图多模态请求，问题: {question[:100]}..., 文件名: {filename}")
    try:
        img_bytes = await image.read()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            IO_EXECUTOR,
            process_single_image_sync,
            img_bytes,
            filename,
            question
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"图片处理失败: {result['error']}")

        return {
            "code": 200,
            "answer": result["answer"],
            "image_description": result["image_description"],
            "docs": result["docs"],
            "agent_steps": result["agent_steps"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("multimodal_rag接口业务异常", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图文问答处理失败: {str(e)}")

@app.post("/api/batch_multimodal_rag")
async def batch_multimodal_rag(
    images: List[UploadFile] = File(...),
    question: str = Form(...)
):
    logger.info(f"接收批量图文请求，图片数量: {len(images)}, 统一问题: {question[:100]}...")
    sem = asyncio.Semaphore(4)

    async def process_single_image(idx: int, upload_img: UploadFile) -> dict:
        async with sem:
            img_bytes = await upload_img.read()
            filename = upload_img.filename or f"image_{idx}.jpg"
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                IO_EXECUTOR,
                process_single_image_sync,
                img_bytes,
                filename,
                question
            )
            result["img_index"] = idx
            return result

    try:
        tasks = [process_single_image(idx, img) for idx, img in enumerate(images)]
        batch_result = await asyncio.gather(*tasks)
        logger.info(f"批量{len(images)}张图片全部处理完成")
        return {
            "code": 200,
            "batch_size": len(images),
            "question": question,
            "data": batch_result
        }
    except Exception as e:
        logger.error("批量图文接口处理异常", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")

if __name__ == "__main__":
    reload_enabled = ENV == "development"
    logger.info(f"启动多模态问答API服务，环境: {ENV}，地址 127.0.0.1:8000，reload: {reload_enabled}")
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=reload_enabled
    )