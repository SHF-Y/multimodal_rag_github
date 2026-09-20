"""reranker 单元测试：本地模型重排 + API 重排 + 边界"""
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from core.reranker import Reranker

@pytest.fixture
def sample_docs():
    return [Document(page_content=f"文档{i}", metadata={}) for i in range(3)]

class TestRerankLocal:
    @patch("sentence_transformers.CrossEncoder")
    def test_sorts_by_score_desc(self, mock_ce, sample_docs):
        """本地重排应按交叉编码器分数降序，且只返回 top_n"""
        mock_ce.return_value.predict.return_value = [0.3, 0.9, 0.6]
        r = Reranker(use_api=False, top_n=2)
        result = r.rerank("查询", sample_docs)
        assert len(result) == 2
        assert result[0].page_content == "文档1"
        assert result[1].page_content == "文档2"
        assert "rerank_score" in result[0].metadata

    @patch("sentence_transformers.CrossEncoder")
    def test_metadata_score_written(self, mock_ce, sample_docs):
        """重排后每个文档都应写入 rerank_score 元数据"""
        mock_ce.return_value.predict.return_value = [0.5, 0.5, 0.5]
        r = Reranker(use_api=False, top_n=2)

        result = r.rerank("查询", sample_docs)
        assert len(result) == 2
        for doc in result:
            assert "rerank_score" in doc.metadata

class TestRerankApi:
    def test_new_format_payload(self, sample_docs):
        """API 模式应按新格式构造 payload（input + parameters）"""
        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.return_value = None
            mock_post.return_value.json.return_value = {
                "output": {"results": [
                    {"index": 2, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.80},
                ]}
            }
            r = Reranker(use_api=True, top_n=2)
            r.api_key = "test-key"
            result = r.rerank("查询", sample_docs)

            payload = mock_post.call_args.kwargs["json"]
            assert payload["model"] == "qwen3-rerank"
            assert payload["input"]["query"] == "查询"
            assert len(payload["input"]["documents"]) == 3
            assert "parameters" in payload

            assert [d.page_content for d in result] == ["文档2", "文档0"]

    def test_legacy_results_fallback(self, sample_docs):
        """兼容旧格式响应（顶层 results）"""
        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.return_value = None
            mock_post.return_value.json.return_value = {
                "results": [{"index": 1, "relevance_score": 0.9}]
            }
            r = Reranker(use_api=True, top_n=2)
            r.api_key = "test-key"

            result = r.rerank("查询", sample_docs)
            assert [d.page_content for d in result] == ["文档1"]

class TestRerankEdge:
    @patch("sentence_transformers.CrossEncoder")
    def test_empty_docs_returns_empty(self, mock_ce):
        """空文档列表直接返回空，不应触发模型"""
        r = Reranker(use_api=False, top_n=3)
        assert r.rerank("查询", []) == []
        mock_ce.return_value.predict.assert_not_called()

    @patch("sentence_transformers.CrossEncoder")
    def test_docs_less_than_topn_no_rerank(self, mock_ce, sample_docs):
        """文档数不超过 top_n 时直接返回原列表，不调用模型"""
        r = Reranker(use_api=False, top_n=5)
        result = r.rerank("查询", sample_docs)
        assert len(result) == 3
        mock_ce.return_value.predict.assert_not_called()
