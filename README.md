# 多模态智能问答系统（工业质检 RAG）

> 面向**变电设备不规则视觉缺陷检测**场景的多模态 RAG 智能问答系统：以 PDF 技术文档为知识库构建混合检索，结合视觉大模型与工具调用，支持纯文本问答、流式输出、单图/多图问答与批量缺陷统计。

## 技术栈

语言   Python 3.12 · Node.js

模型平台   阿里云百炼 DashScope（通义千问）· 对话 qwen3.7-plus · 视觉 qwen-vl-plus · 向量 text-embedding-v4 · 在线精排 qwen3-rerank

RAG   langchain-community / langchain-core / langchain-openai / langchain-text-splitters · rank_bm25 · jieba · pypdf · sentence-transformers（本地 CrossEncoder 精排）

Web 框架   FastAPI · Uvicorn · Gradio

存储   ChromaDB · Redis 7

前端   Vue 3 · Vite 5 · marked · @vitejs/plugin-vue

部署   Docker · docker-compose · Nginx

测试/评测   pytest · ragas · datasets 

基础库   python-dotenv · tenacity · httpx · requests · Pillow

## 目录结构

```
multimodal_rag/
├── api_llm_pro_latest.py        # 后端主入口
├── gradio_app.py                # Gradio 演示界面
├── conftest.py                  # pytest 全局 fixture
├── pytest.ini                   # pytest 配置
├── requirements.txt             # Python 依赖
├── Dockerfile                   # 多阶段构建
├── .env                         # 环境变量
│
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── cache_manager.py         #   Redis / 内存双缓存（自动降级）
│   ├── config.py                #   配置常量 + LLM / Embedding 
│   ├── hybrid_retriever.py      #   向量 + BM25 混合检索
│   ├── query_rewriter.py        #   查询改写
│   ├── rag_module.py            #   向量库初始化、文档导出、检索器单例、format_docs
│   ├── rate_limiter.py          #   令牌桶限流 + 熔断器
│   ├── reranker.py              #   精排（API / 本地 CrossEncoder）
│   ├── session_manager.py       #   会话管理（滑动窗口 + 摘要）
│   ├── stream_agent.py          #   流式 Agent（工具调用循环 + SSE 事件）
│   ├── tools_module.py          #   业务工具（文字提取 / 缺陷识别 / 批量统计 + ALL_TOOLS）
│   ├── vectorstore_manager.py   #   向量库构建/加载、父子分块检索
│   └── vision_module.py         #   视觉大模型调用（tenacity 重试）
│
├── src/                         # Vue3 前端
│   ├── main.js
│   ├── App.vue
│   └── compoents/ChatInterface.vue
│
├── data/
│   ├── pdfs/                    # PDF 知识库
│   └── images/                  # 图片样例
│
├── tests/                       # pytest 测试
│   ├── test_api.py              #   API 集成测试
│   ├── test_rag_module.py       #   RAG 模块（含 format_docs / 检索器单例）
│   ├── test_hybrid_retriever.py #   混合检索
│   ├── test_reranker.py         #   精排
│   ├── test_query_rewriter.py   #   查询改写
│   ├── test_rate_limiter.py     #   限流 / 熔断
│   ├── test_cache_manager.py    #   缓存
│   ├── test_session_manager.py  #   会话
│   ├── test_stream_agent.py     #   流式 Agent
│   ├── test_vision_module.py    #   视觉模块
│   ├── test_vectorstore_manager.py  # 向量库管理
│   └── test_tools_module.py     #   工具模块
│
├── evaluation/                  # 离线评测（检索 / RAG / 视觉）
│   ├── __init__.py
│   ├── run_evaluation.py        #   评测入口
│   ├── retrieval_eval.py        #   检索质量评测
│   ├── rag_evaluator.py         #   RAG 问答评测
│   ├── dataset_builder.py       #   评测数据集构建
│   └── test_cases1.json         #   评测用例数据
|
├── chroma_db/                   # 向量库持久化目录
├── logs/                        # 日志目录
└── redis-server/                # Windows 本地 Redis 运行包
```

---

### 方式一：本地运行

**1. 安装依赖**

```bash
pip install -r requirements.txt
```

**2. 配置环境变量**

直接创建 `.env`，填入 API_KEY

**3. 启动后端**

```bash
python api_llm_pro_latest.py
```

API 服务：http://127.0.0.1:8000 ，接口文档（Swagger）：http://127.0.0.1:8000/docs


**4. 启动前端**

```bash
npm install
npm run dev              
```
![图片](<前端界面截图.jpg>)
### 方式二：Docker 一键部署

```bash
docker compose up -d --build
```

---






