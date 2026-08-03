
import logging
import hashlib
import time
import asyncio

from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request


from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from typing import List, Dict, Any, Optional
from collections import OrderedDict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool

from core.config import get_llm
from core.rag_module import get_retriever, format_docs, init_vector_store
from core.vision_module import parse_image
from core.tools_module import ALL_TOOLS

# ====================== 1. 日志工程化配置 ======================
#定义日志输出格式：时间 - 日志级别 - 记录器名称 - 消息内容。
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S"
)
#logger
logger = logging.getLogger("IndustrialMultiModalAPI")

# ====================== 2. 全局性能缓存配置 ======================

IMAGE_CACHE: Dict[str, Dict[str, Any]] = {}#


RAG_CACHE: Dict[str, Dict[str, Any]] = {}

CACHE_EXPIRE_MINUTES = 10       # 缓存过期时间：10分钟
MAX_TOOL_LOOP = 5               # 工具循环最大轮次常量
TMP_FILE_EXPIRE = 3600          # 临时文件过期清理阈值（秒）
TMP_FILE_RECORD: Dict[str, float] = {}  


LAST_CLEAN_TIME: float = 0.0
CLEAN_INTERVAL: int = 60       

# ====================== FastAPI 初始化 ======================
app = FastAPI(title="多模态问答系统API ")
# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== 全局异常捕获中间件 ======================
@app.middleware("http")
async def global_exception_handler(request: Request, call_next):#

    start_time = time.time()
    try:#正常流程
        response = await call_next(request)
        cost = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Request {request.method} {request.url.path} | cost {cost}ms")
        return response
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        cost = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Request {request.method} {request.url.path} failed, cost {cost}ms, error: {str(e)}", exc_info=True)
        err_resp = {
            "code": 500,
            "msg": f"服务内部异常: {str(e)}",
            "data": None
        }
        return JSONResponse(content=err_resp, status_code=500)

# ====================== 全局资源初始化 ======================
try:
    logger.info("开始初始化向量库与大模型资源...")#记录开始日志。
    init_vector_store()
    llm_raw = get_llm()
    retriever = get_retriever()
    # 绑定工具到LLM
    llm_with_tools = llm_raw.bind_tools(ALL_TOOLS)
    tool_map: Dict[str, Tool] = {tool.name: tool for tool in ALL_TOOLS}
    logger.info("向量库、LLM、工具资源初始化完成")
except Exception as init_err:
    logger.critical("全局资源初始化失败，服务无法启动", exc_info=True) 
    raise RuntimeError("核心资源初始化异常") from init_err

# ====================== 缓存与临时文件清理函数 ======================
def clean_expire_cache():
    """全量清理过期图片缓存、RAG缓存、临时文件，增加最小间隔限制"""
    global LAST_CLEAN_TIME   
    now = time.time()
    if now - LAST_CLEAN_TIME < CLEAN_INTERVAL:
        return
    LAST_CLEAN_TIME = now

    expire_ts = now - CACHE_EXPIRE_MINUTES * 60
    del_keys = [k for k, v in IMAGE_CACHE.items() if v["expire"] < expire_ts]
    for k in del_keys:
        del IMAGE_CACHE[k]#遍历 IMAGE_CACHE，找出过期条目的键，然后删除。
    if del_keys:
        logger.info(f"清理过期图片缓存 {len(del_keys)} 条")

    del_rag_keys = [k for k, v in RAG_CACHE.items() if v["expire"] < expire_ts]
    for k in del_rag_keys:
        del RAG_CACHE[k]
    if del_rag_keys:
        logger.info(f"清理过期RAG缓存 {len(del_rag_keys)} 条")

    # 清理过期临时文件
    tmp_del = [path for path, create_ts in TMP_FILE_RECORD.items() if create_ts< now - TMP_FILE_EXPIRE]
    for path in tmp_del:
        if os.path.exists(path):
            try:
                os.unlink(path)
                logger.warning(f"自动清理过期临时图片文件: {path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败 {path}: {str(e)}")
        del TMP_FILE_RECORD[path]

def calc_file_md5(file_bytes: bytes) -> str:
    """计算文件md5用于图片去重"""
    md5_obj = hashlib.md5()
    md5_obj.update(file_bytes)
    return md5_obj.hexdigest()

# ====================== 图片缓存 ======================
def get_cached_image_desc(file_md5: str, prompt: str) -> Optional[str]:
   
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    cache_key = f"{file_md5}:{prompt_hash}"
 
    now = time.time()
    item = IMAGE_CACHE.get(cache_key)
    if not item:
        return None
    if item["expire"] < now:
        del IMAGE_CACHE[file_md5]
        return None
    return item["desc"]

def set_image_cache(file_md5: str, prompt: str, desc: str):
    """写入图片缓存，关联解析Prompt"""
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    cache_key = f"{file_md5}:{prompt_hash}"
    expire = time.time() + CACHE_EXPIRE_MINUTES * 60
    IMAGE_CACHE[cache_key] = {"desc": desc, "expire": expire}

# ====================== RAG缓存 ======================
def get_cached_rag_docs(query: str) -> Optional[List[Any]]:
    """获取RAG缓存结果，惰性过期检查"""
    now = time.time()
    item = RAG_CACHE.get(query)
    if not item:
        return None
    if item["expire"] < now:
        del RAG_CACHE[query]
        return None
    return item["docs"]

def set_rag_cache(query: str, docs: List[Any]):
    """写入RAG缓存，设置过期时间"""
    expire = time.time() + CACHE_EXPIRE_MINUTES * 60
    RAG_CACHE[query] = {"docs": docs, "expire": expire}

# ========== 工具调用==============================
def run_llm_with_manual_tools(input_messages: List[HumanMessage]) -> tuple[str, List[dict]]:
    """
    :input_messages: 初始用户消息列表
    :return: (最终回答文本, 工具执行步骤记录)
    """
    messages = input_messages.copy()
    tool_steps = []
    loop_count = 0
    logger.info(f"启动LLM工具循环，最大轮次 {MAX_TOOL_LOOP}")

    while loop_count < MAX_TOOL_LOOP:
        loop_count += 1
        logger.info(f"===== 工具循环第 {loop_count} 轮 =====")#记录当前轮次。
        try:
            # 1. 调用绑定工具的LLM，获取模型输出
            ai_msg: AIMessage = llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            logger.info(f"LLM返回内容片段: {ai_msg.content[:200]}...")
        except Exception as llm_err:#
            logger.error(f"第{loop_count}轮LLM调用失败", exc_info=True)
            return f"大模型调用异常：{str(llm_err)}", tool_steps

        # 2. 判断是否存在工具调用
        if not ai_msg.tool_calls:
            logger.info("LLM无工具调用，结束循环")
            break

        # 3. 遍历所有工具调用，逐个执行
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            call_id = tool_call["id"]
            logger.info(f"准备调用工具 {tool_name}, 参数: {tool_args}, call_id: {call_id}")
            tool_steps.append({
                "round": loop_count,
                "step": "tool_call",
                "tool": tool_name,
                "args": tool_args,
                "id": call_id
            })
            # 执行工具
            try:
                if tool_name not in tool_map:
                    tool_result = f"错误：不存在名为 {tool_name} 的工具"
                    logger.warning(tool_result)
                else:
                    tool = tool_map[tool_name]
                    tool_result = tool.invoke(tool_args)
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
   
    final_ai_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_ai_msg = msg
            break

    if final_ai_msg is None:
        final_answer = "模型未返回有效回答"
    else:
        final_answer = final_ai_msg.content if hasattr(final_ai_msg, "content") else str(final_ai_msg)

    if loop_count >= MAX_TOOL_LOOP and final_ai_msg and final_ai_msg.tool_calls:
        final_answer = f"【提示：已达到最大工具调用轮次({MAX_TOOL_LOOP}轮)，回答可能不完整】\n{final_answer}"
        logger.warning(f"工具循环达到最大轮次{MAX_TOOL_LOOP}，存在未执行的工具调用")

    logger.info(f"工具循环结束，生成最终回答长度: {len(final_answer)}")
    return final_answer, tool_steps

# ========== 纯文本RAG接口 ==========
@app.post("/api/text_rag")
def text_rag_query(question: str = Form(...)):
    """纯文本RAG问答"""
    logger.info(f"接收文本RAG请求，问题: {question}")
    try:
        # 1. RAG检索，优先读缓存
        docs = get_cached_rag_docs(question)
        if docs is None:
            logger.info("未命中RAG缓存，执行向量库检索")
            docs = retriever.invoke(question)
            set_rag_cache(question, docs)
        context = format_docs(docs)

        system_rule = """你是一名严格遵守规则的工业质检专家。你的所有结论必须基于：
  - 工具返回的实际结果
  - 知识库中的参考文档
  - 用户提供的信息
你绝不能编造数据、猜测统计结果或给出你无法核实的建议。
【工具使用优先级与场景】
1. 批量缺陷统计：当用户要求统计某个文件夹下缺陷，优先使用批量缺陷统计工具。
2. 图片分析：涉及图片识别、缺陷、OCR时使用对应图片工具。
3. 知识库问答：无工具需求时直接基于文档回答，禁止主观猜测。
工具缺失路径时主动向用户索要；工具无结果如实告知，禁止模糊推测。
"""
        user_input = f"{system_rule}\n知识库参考文档：\n{context}\n\n用户问题：{question}"
        input_msg = [HumanMessage(content=user_input)]
        answer, tool_steps = run_llm_with_manual_tools(input_msg)

        resp = {
            "code": 200,
            "answer": answer,
            "docs": [d.page_content for d in docs],
            "agent_steps": tool_steps
        }
        return resp
    except Exception as e:
        logger.error(f"text_rag接口业务异常", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文本问答处理失败: {str(e)}")

# ========== 单张图文接口 ==========
@app.post("/api/multimodal_rag")
async def multimodal_rag_query(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    """单张图片图文问答"""
    logger.info(f"接收单图多模态请求，问题: {question}, 文件名: {image.filename}")
    tmp_path = ""
    try:
        img_bytes = await image.read()
        suffix = os.path.splitext(image.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
            TMP_FILE_RECORD[tmp_path] = time.time()
        logger.info(f"临时图片保存路径: {tmp_path}")
        file_md5 = calc_file_md5(img_bytes)
        logger.info(f"上传图片md5: {file_md5}")
        parse_prompt = "详细描述图片中零件的外观、缺陷特征"
        img_desc = get_cached_image_desc(file_md5, parse_prompt)
        if img_desc is not None:
            logger.info(f"图片md5 {file_md5} 命中缓存，跳过视觉模型推理")
        else:
            logger.info("未命中图片缓存，执行视觉解析")
            img_desc = await asyncio.to_thread(parse_image, tmp_path, parse_prompt)
            set_image_cache(file_md5, parse_prompt, img_desc)

        search_query = f"{img_desc}\n{question}"
        docs = get_cached_rag_docs(search_query)
        if docs is None:
            docs = await asyncio.to_thread(retriever.invoke, search_query)
            set_rag_cache(search_query, docs)
        context = format_docs(docs)

        system_rule = """你是一名严格遵守规则的工业质检专家。你的所有结论必须基于：
  - 工具返回的实际结果
  - 知识库中的参考文档
  - 用户提供的信息
你绝不能编造数据、猜测统计结果或给出你无法核实的建议。
可使用工具读取临时图片路径下的图片、识别缺陷、提取文字。
工具缺失路径时主动向用户索要；工具无结果如实告知，禁止模糊推测。
"""
        user_input = f"""{system_rule}
知识库参考文档：{context}
图片本地临时路径：{tmp_path}
图片基础描述：{img_desc}
用户问题：{question}
"""
        input_msg = [HumanMessage(content=user_input)]

        answer, tool_steps = await asyncio.to_thread(run_llm_with_manual_tools, input_msg)
        
        resp = {
            "code": 200,
            "answer": answer,
            "image_description": img_desc,
            "docs": [d.page_content for d in docs],
            "agent_steps": tool_steps
        }
        logger.info("单图多模态接口处理完成")
        return resp
    except Exception as e:
        logger.error("multimodal_rag接口业务异常", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图文问答处理失败: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)#
                TMP_FILE_RECORD.pop(tmp_path, None)#
                logger.info(f"已清理临时图片: {tmp_path}")
            except Exception as clean_e:
                logger.warning(f"临时文件删除失败 {tmp_path}: {str(clean_e)}")

# ========== 批量接口 ==========
@app.post("/api/batch_multimodal_rag")
async def batch_multimodal_rag(
    images: List[UploadFile] = File(...),
    question: str=Form(...)
    ):
    """批量上传接口，处理多张图片"""
    logger.info(f"接收批量图文请求，图片数量: {len(images)}, 统一问题: {question}")
    tmp_path_list = []

    async def process_single_image(idx: int, upload_img: UploadFile) -> dict:
        """定义一个单张图片处理协程"""
        img_bytes = await upload_img.read()
        suffix = os.path.splitext(upload_img.filename)[1] or ".jpg"

      
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
            tmp_path_list.append(tmp_path)
            TMP_FILE_RECORD[tmp_path] = time.time()

        try:
            file_md5 = calc_file_md5(img_bytes)
            parse_prompt = "详描述图片中零件的外观、缺陷特征"

           
            img_desc = get_cached_image_desc(file_md5, parse_prompt)
            if img_desc is None:
                img_desc = await asyncio.to_thread(parse_image, tmp_path, parse_prompt)
                set_image_cache(file_md5, parse_prompt, img_desc)
            
            search_query = f"{img_desc}\n{question}"
            docs = get_cached_rag_docs(search_query)
            if docs is None:
                docs = await asyncio.to_thread(retriever.invoke, search_query)#
                set_rag_cache(search_query, docs)
            context = format_docs(docs)

            system_rule = """你是一名严格遵守规则的工业质检专家。你的所有结论必须基于：
  - 工具返回的实际结果
  - 知识库中的参考文档
  - 用户提供的信息
你绝不能编造数据、猜测统计结果或给出你无法核实的建议。
可使用工具读取临时图片路径下的图片、识别缺陷、提取文字。
工具缺失路径时主动向用户索要；工具无结果如实告知，禁止模糊推测。
"""
            user_input = f"""{system_rule}
知识库参考文档：{context}
图片本地临时路径：{tmp_path}
图片基础描述：{img_desc}
用户问题：{question}
"""
            input_msg = [HumanMessage(content=user_input)]
            answer, tool_steps = await asyncio.to_thread(run_llm_with_manual_tools, input_msg)
            return {
                "img_index": idx,
                "filename": upload_img.filename,
                "img_md5": file_md5,
                "image_description": img_desc,
                "answer": answer,
                "docs": [d.page_content for d in docs],
                "agent_steps": tool_steps
            }
        except Exception as e:
            logger.error(f"批量处理第{idx+1}张图失败: {upload_img.filename}", exc_info=True)
            
            return {
                "img_index": idx,
                "filename": upload_img.filename,
                "error": str(e),
                "answer": None,
                "agent_steps": []
            }

    try:
     
        tasks = [process_single_image(idx, img) for idx, img in enumerate(images)]
        batch_result = await asyncio.gather(*tasks)#
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
    finally:
        for p in tmp_path_list:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                    TMP_FILE_RECORD.pop(p, None)
                except Exception as e:
                    logger.warning(f"批量清理临时文件失败 {p}: {str(e)}")
        clean_expire_cache()

if __name__ == "__main__":
    logger.info("启动多模态问答API服务，地址 127.0.0.1:8000")
    uvicorn.run("api_tools_llm_pro:app", host="127.0.0.1", port=8000, reload=True)

# 测试文件夹路径：批量统计文件夹缺陷类型 f:/multimodal_rag/data/images2/
