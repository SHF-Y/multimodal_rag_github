"""
Reranker重排序模块：使用交叉编码器对检索结果做精排
方案选择：
- 本地模型：BAAI/bge-reranker-v2-m3（中英文通用）
- 在线API：阿里云的rerank API（避免本地加载大模型）
这里支持两种模式，通过配置切换
"""
import os
from typing import List
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 use_api: bool = False, top_n: int = 3):
        self.top_n = top_n
        self.use_api = use_api

        if use_api:

            self.api_key = os.getenv("DASHSCOPE_API_KEY")

            self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

        else:

            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)

    def _rerank_local(self, query: str, docs: List[Document],top_n) -> List[Document]:
        """本地模型重排序"""

        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in scored_docs[:top_n]:
            doc.metadata["rerank_score"] = float(score)
            result.append(doc)
        return result

    def _rerank_api(self, query: str, docs: List[Document],top_n) -> List[Document]:
        """在线API重排序（新格式：input + parameters 嵌套）"""
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "qwen3-rerank",
            "input": {
                "query": query,
                "documents": [doc.page_content for doc in docs]
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False
            }
        }
        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("output", {}).get("results") or data.get("results", [])

        reranked = []
        for item in results:
            idx = item["index"]
            score = item["relevance_score"]
            docs[idx].metadata["rerank_score"] = score
            reranked.append(docs[idx])
        return reranked

    def rerank(self, query: str, docs: List[Document],top_n=None) -> List[Document]:
        """对外统一重排序接口,top_n 不传时用 __init__ 的默认值"""
        if not docs:
            return []
        n = top_n if top_n is not None else self.top_n
        if len(docs) <= n:
            return docs
        if self.use_api:
            return self._rerank_api(query, docs,n)
        return self._rerank_local(query, docs, n)