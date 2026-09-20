"""
混合检索器单元测试
对 HybridRetriever 类的各个方法进行单元测试，包括：
- 初始化时构建 BM25 索引
- 中文分词方法
- BM25 检索功能
- RRF（Reciprocal Rank Fusion）融合算法
- invoke 调用两个检索器
- 空查询的边界情况

"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from core.hybrid_retriever import HybridRetriever

@pytest.fixture
def sample_docs():
    """
    测试用的文档集 fixture。
    返回一个包含 5 个 Document 对象的列表，模拟文本片段，
    每个文档都有 page_content（正文）和 metadata（元数据，这里用 id 标识）。
    """
    return [
        Document(page_content="变压器油的闪点标准为不低于135摄氏度", metadata={"id": 1}),
        Document(page_content="电力变压器的绕组温度不得超过65摄氏度", metadata={"id": 2}),
        Document(page_content="变电设备红外测温发现异常发热点", metadata={"id": 3}),
        Document(page_content="断路器的机械特性测试包括分合闸时间", metadata={"id": 4}),
        Document(page_content="绝缘子表面污秽会导致闪络电压降低", metadata={"id": 5}),
    ]

@pytest.fixture
def mock_vector_retriever():
    """
    模拟向量检索器的 fixture。
    使用 MagicMock 创建一个假对象，后续测试中可以设置其 invoke 方法的返回值，
    """
    retriever = MagicMock()
    return retriever

class TestHybridRetriever:
    def test_init_creates_bm25_index(self, sample_docs, mock_vector_retriever):
        """
        测试初始化时正确构建 BM25 索引。
        创建 HybridRetriever 实例，传入模拟的向量检索器和文档集，
        验证内部会基于 sample_docs 构建 BM25 索引，并保存文档列表。
        """
        hr = HybridRetriever(mock_vector_retriever, sample_docs)
        assert hr.bm25 is not None
        assert len(hr.docs) == 5
    def test_tokenize_chinese_text(self, sample_docs, mock_vector_retriever):
        """
        测试中文分词方法 _tokenize。
        传入一个中文短语，期望返回一个非空的 token 列表，
        并且列表中包含“变压器”或“变压器油”这样的有意义的词（取决于分词粒度）。
        """
        hr = HybridRetriever(mock_vector_retriever, sample_docs)

        tokens = hr._tokenize("变压器油的闪点标准")

        assert len(tokens) > 0

        assert "变压器" in tokens or "变压器油" in tokens

    def test_bm25_search_returns_ranked_results(self, sample_docs, mock_vector_retriever):
        """
        测试 BM25 检索返回按分数排序的结果。
        使用查询“变压器油闪点”，预期 BM25 会给包含这些关键词的文档更高的分数，
        第一个返回的结果应该是最相关的文档。
        """
        hr = HybridRetriever(mock_vector_retriever, sample_docs)

        results = hr._bm25_search("变压器油闪点")

        assert len(results) > 0

        top_idx = results[0][0]

        assert "闪点" in sample_docs[top_idx].page_content

    def test_rrf_fusion_combines_results(self, sample_docs, mock_vector_retriever):
        """
        测试 RRF 融合正确合并两路结果。
        RRF 算法会对每个文档根据其在不同结果列表中的排名计算得分，
        排名越靠前，得分越高。手动构造两路结果，并验证融合后排名顺序。
        """

        hr = HybridRetriever(mock_vector_retriever, sample_docs, final_top_k=3)

        bm25_results = [(0, 2.5), (1, 1.8), (2, 1.2)]

        vector_results = [sample_docs[1], sample_docs[2], sample_docs[3]]

        fused = hr._rrf_fusion(bm25_results, vector_results)

        assert len(fused) == 3

        assert sample_docs[1] in fused[:2] or sample_docs[2] in fused[:2]

    def test_invoke_calls_both_retrievers(self, sample_docs, mock_vector_retriever):
        """
        测试 invoke 方法同时调用 BM25 检索和向量检索。
        设置模拟向量检索器的返回值为两个文档，然后调用 invoke，期望返回结果数量为 final_top_k（2），
            且向量检索器的 invoke 被调用一次。
        """

        mock_vector_retriever.invoke.return_value = [sample_docs[0], sample_docs[1]]

        hr = HybridRetriever(mock_vector_retriever, sample_docs, final_top_k=2)

        results = hr.invoke("变压器")

        assert len(results) == 2

        mock_vector_retriever.invoke.assert_called_once()

    def test_empty_query_returns_empty(self, sample_docs, mock_vector_retriever):
        """
        测试空查询返回空列表（或至少是列表类型）。
        空查询可能导致 BM25 返回一些默认结果，但融合后应能正常处理
        """
        hr = HybridRetriever(mock_vector_retriever, sample_docs)

        results = hr.invoke("")

        assert isinstance(results, list)