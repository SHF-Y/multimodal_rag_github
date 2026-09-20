"""
混合检索器：BM25关键词检索 + 向量语义检索，加权融合后返回Top-K
"""
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Any, Optional
from langchain_core.documents import Document

class HybridRetriever:
    def __init__(self, vector_retriever, docs: List[Document],
                 bm25_top_k: int = 10, vector_top_k: int = 10,
                 final_top_k: int = 5, rrf_k: int = 60):
        """
        Args:
            vector_retriever: 已经创建好的向量检索器，比如 Chroma 的检索器。它负责向量检索。
            docs: 所有文档列表，用于构建BM25索引
            bm25_top_k: BM25单路召回数量
            vector_top_k: 向量检索单路召回数量
            final_top_k: 融合后最终返回数量
            rrf_k: RRF融合参数，控制排名衰减速度
        """
        self.vector_retriever = vector_retriever
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

        self.docs = docs
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """中文分词：精确模式，过滤单字和空白"""
        tokens = jieba.lcut(text)

        return [t.strip() for t in tokens if len(t.strip()) > 1]

    def _bm25_search(self, query: str) -> List[tuple]:
        """BM25检索，返回 [(doc_index, score), ...] 按分数降序"""
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:self.bm25_top_k]

    def _vector_search(self, query: str) -> List[Document]:
        """向量检索，返回文档列表（限制为 vector_top_k 条）"""
        results = self.vector_retriever.invoke(query)
        return results[:self.vector_top_k]

    def _rrf_fusion(self, bm25_results: List[tuple],
                     vector_results: List[Document]) -> List[Document]:
        """
        Reciprocal Rank Fusion 融合算法
        公式：score = 1 / (k + rank)
        优势：不需要归一化不同检索器的分数尺度
        """
        fusion_scores = {}

        for rank, (doc_idx, _) in enumerate(bm25_results):
            fusion_scores[doc_idx] = fusion_scores.get(doc_idx, 0) + 1.0 / (self.rrf_k + rank)

        for rank, doc in enumerate(vector_results):

            doc_idx = self._find_doc_index(doc)
            if doc_idx is not None:
                fusion_scores[doc_idx] = fusion_scores.get(doc_idx, 0) + 1.0 / (self.rrf_k + rank)

        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.docs[idx] for idx, _ in sorted_docs[:self.final_top_k]]

    def _find_doc_index(self, target_doc: Document) -> Optional[int]:
        """通过内容匹配查找文档索引（建议用doc_id）"""
        for i, doc in enumerate(self.docs):
            if doc.page_content == target_doc.page_content:
                return i
        return None

    def invoke(self, query: str) -> List[Document]:
        """对外统一检索接口"""
        bm25_results = self._bm25_search(query)
        vector_results = self._vector_search(query)
        return self._rrf_fusion(bm25_results, vector_results)