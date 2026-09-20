

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
from unittest.mock import patch, MagicMock

from concurrent.futures import ThreadPoolExecutor, Future

@pytest.fixture(autouse=True)
def mock_global_resources():
    """
    自动 mock 所有全局资源。
    包括：LLM_EXECUTOR、IO_EXECUTOR、retriever、reranker、session_manager 等。
    这些全局资源通常在 FastAPI 应用的启动事件（lifespan）中初始化，
    但在测试环境中，并不启动 lifespan，因此这些变量会保持为 None，导致调用时出错。
    所以在测试前用模拟对象替换它们，保证测试可以正常运行。
    """

    mock_executor = MagicMock(spec=ThreadPoolExecutor)

    def mock_submit(fn, *args, **kwargs):

        future = Future()
        try:

            result = fn(*args, **kwargs)

            future.set_result(result)
        except Exception as e:

            future.set_exception(e)
        return future

    mock_executor.submit.side_effect = mock_submit

    mock_retriever = MagicMock()

    mock_retriever.invoke.return_value = [
        MagicMock(page_content="变压器油闪点不低于135度，闭口杯法测定", metadata={})

    ]

    with patch("api_llm_pro_latest.LLM_EXECUTOR", mock_executor), \
         patch("api_llm_pro_latest.IO_EXECUTOR", mock_executor), \
         patch("api_llm_pro_latest.retriever", mock_retriever), \
         patch("api_llm_pro_latest.reranker", MagicMock()) as mock_reranker, \
         patch("api_llm_pro_latest.session_manager") as mock_session_manager:

        mock_reranker.rerank.side_effect = lambda query, docs, top_n=3: docs[:top_n]

        mock_session_manager.get_or_create.return_value = MagicMock()

        yield

