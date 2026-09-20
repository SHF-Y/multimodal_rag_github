
"""
被测模块：core/rag_module.py
被测函数：
     format_docs —— 把检索到的文档列表格式化成"喂给 LLM 的上下文文本"
     init_vector_store —— 初始化向量库（拿 embedding + 加载/构建）
     _get_all_docs     —— 从向量库导出全部文档，供 BM25 索引构建
     get_retriever     —— 获取混合检索器
"""
from langchain_core.documents import Document

import pytest

import core.rag_module as rag_module

from unittest.mock import MagicMock, patch

class TestFormatDocs:
    """
    format_docs 格式化契约测试组
    ------------------------------------------------------------------------
    输入 Document 列表，输出按 "[文档N]: 内容" 编号、 并以空行分隔的纯文本，供 LLM 作为检索上下文使用。
    """

    def test_formats_multiple_docs(self):
        """多文档时按顺序编号 "[文档1]/[文档2]" 并保留全文内容。"""
        docs = [
            Document(page_content="锈蚀缺陷分为轻中重三级"),
            Document(page_content="渗漏油缺陷按形态分类"),
        ]
        out =rag_module.format_docs(docs)

        assert "[文档1]: 锈蚀缺陷分为轻中重三级" in out

        assert "[文档2]: 渗漏油缺陷按形态分类" in out

    def test_empty_docs(self):
        """空文档列表应返回空字符串"""
        assert rag_module.format_docs([]) == ""

    def test_single_doc(self):
        """单文档也按统一格式输出（编号从 [文档1] 开始）。"""
        out = rag_module.format_docs([Document(page_content="只有一条")])
        assert out == "[文档1]: 只有一条"

    def test_newline_separator(self):
        """多条文档之间必须用空行（\\n\\n）分隔，让 LLM 能识别"这是两条独立文档""""
        docs = [Document(page_content="A"), Document(page_content="B")]
        out = rag_module.format_docs(docs)
        assert "\n\n" in out

@pytest.fixture(autouse=True)
def reset_global_cache():
    """每个测试前后重置 rag_module 的两个全局缓存（_all_docs / _hybrid_retriever）。"""

    rag_module._all_docs = None
    rag_module._hybrid_retriever = None

    yield

    rag_module._all_docs = None
    rag_module._hybrid_retriever = None

class TestInitVectorStore:
    """init_vector_store 函数测试
    测试要点：它自己不做任何业务逻辑，只是"组装"两步调用.
    所以断言"第二步是否收到了正确的三个参数"。
    """

    def test_returns_vectorstore(self):
        """ get_embedding() 的返回值应传给 load_or_create_vectorstore，
        并使用配置中的向量库目录与 PDF 目录。"""
        mock_embedding = MagicMock()

        mock_vs = MagicMock()

        with patch("core.rag_module.get_embedding", return_value=mock_embedding), \
             patch("core.rag_module.load_or_create_vectorstore", return_value=mock_vs) as mock_load:

            result = rag_module.init_vector_store()

        assert result is mock_vs

        mock_load.assert_called_once_with(
            rag_module.VECTOR_DB_PATH, rag_module.PDF_FOLDER, mock_embedding
        )

class TestGetAllDocs:
    """_get_all_docs 函数测试组
    ------------------------------------------------------------------------
        - 文档列表能否从原始 documents/metadatas 正确重建
        - metadata 为 None 时能否优雅降级为 {}
        - 缓存是否真的生效（第二次调用不再初始化向量库）
    """

    def test_builds_docs_from_collection(self):
        """验证从 vectorstore._collection.get 的 documents/metadatas 重建 Document 列表，
        且 metadata 为空（None）时降级为 {}。"""

        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {
            "documents": ["文档A", "文档B"],
            "metadatas": [{"source": "a.pdf"}, None],
        }

        with patch("core.rag_module.init_vector_store", return_value=mock_vs):
            docs = rag_module._get_all_docs()

        assert len(docs) == 2

        assert isinstance(docs[0], Document)

        assert docs[0].page_content == "文档A"
        assert docs[0].metadata == {"source": "a.pdf"}

        assert docs[1].metadata == {}

    def test_caches_after_first_call(self):
        """第二次调用应命中全局缓存，不再重复初始化向量库。
        真实场景里每次请求都重建 BM25 索引会非常慢，所以源码用全局缓存保证"整个进程只初始化一次"。"""
        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {
            "documents": ["唯一文档"],
            "metadatas": [{}],
        }
        with patch("core.rag_module.init_vector_store", return_value=mock_vs) as mock_init:
            docs1 = rag_module._get_all_docs()
            docs2 = rag_module._get_all_docs()

        assert docs1 is docs2
        mock_init.assert_called_once()

class TestGetRetriever:
    """get_retriever 函数测试组
    -----------------------------------------------------------------------
        - HybridRetriever 的构造参数是否符合设计（BM25/向量各召回 10，最终取 k）
        - 第二次调用不再构造新实例，只更新 final_top_k
    """

    def test_creates_hybrid_retriever_once_and_updates_k(self):
        """首次调用构造 HybridRetriever（含向量检索器、BM25 文档、top_k 参数），
        第二次调用复用同一实例，仅更新 final_top_k。"""

        mock_vs = MagicMock()
        mock_vector_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_vector_retriever

        mock_docs = [Document(page_content="构建BM25索引用文档")]

        mock_hybrid = MagicMock()

        with patch("core.rag_module.init_vector_store", return_value=mock_vs), \
             patch("core.rag_module._get_all_docs", return_value=mock_docs), \
             patch("core.rag_module.HybridRetriever", return_value=mock_hybrid) as mock_hr_cls:

            r1 = rag_module.get_retriever(k=5)

            r2 = rag_module.get_retriever(k=10)

        assert r1 is mock_hybrid
        assert r2 is mock_hybrid

        mock_hr_cls.assert_called_once_with(
            vector_retriever=mock_vector_retriever,
            docs=mock_docs,
            bm25_top_k=10,
            vector_top_k=10,
            final_top_k=5,
        )

        assert mock_hybrid.final_top_k == 10

        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 10})
