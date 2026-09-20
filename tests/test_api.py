import pytest

from httpx import AsyncClient, ASGITransport

from unittest.mock import patch, MagicMock

from api_llm_pro_latest import app

@pytest.fixture
def mock_llm():


    with patch("api_llm_pro_latest.get_thread_llm_plain") as mock_plain, \
         patch("api_llm_pro_latest.query_rewriter") as mock_rewriter:

        mock_llm_instance = MagicMock()

        mock_llm_instance.invoke.return_value = MagicMock(
            content="变压器油的闪点标准是不低于135摄氏度。",
            tool_calls=[])

        mock_plain.return_value = mock_llm_instance

        mock_rewriter.rewrite.return_value = ["变压器油闪点标准是多少？"]
        yield mock_plain

@pytest.mark.asyncio
async def test_health_endpoint():


    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:

        response = await ac.get("/health")

    assert response.status_code == 200

@pytest.mark.asyncio
async def test_chat_endpoint(mock_llm):


    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:

        response = await ac.post(
            "/api/chat",
            files={
                "question": (None, "变压器油闪点标准是多少？"),
                "session_id": (None, "test_sess_001"),
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["code"] == 200

    assert "answer" in data

    assert "docs" in data

    assert data["session_id"] == "test_sess_001"

@pytest.mark.asyncio
async def test_chat_missing_param():


    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:

        response = await ac.post("/api/chat", files={"session_id": (None, "s1")})

    assert response.status_code == 422

@pytest.fixture
def mock_vision():


    with patch("api_llm_pro_latest.parse_image") as mock_parse, \
         patch("api_llm_pro_latest.cache_manager") as mock_cache:

        mock_parse.return_value = "这是一张变电设备红外测温图，设备整体外观正常。"

        mock_cache.get_or_set.side_effect = (
            lambda key, loader_func, ttl=None, use_lock=False: loader_func())

        yield

@pytest.mark.asyncio
async def test_chat_multimodal(mock_llm, mock_vision):


    import io

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/chat",
            data={"question": "这张图片有什么缺陷？", "session_id": "mm_sess_001"},
            files={"images": ("test.png", io.BytesIO(png_bytes), "image/png")})


    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert data["mode"] == "multimodal"
    assert len(data["multimodalResults"]) == 1
    assert data["multimodalResults"][0]["answer"]
    assert "image_description" in data["multimodalResults"][0]

@pytest.mark.asyncio
async def test_chat_multimodal_multi_images(mock_llm, mock_vision):


    import io
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = [
            ("images", (f"test_{i}.png", io.BytesIO(png_bytes), "image/png"))
            for i in range(3)]
        response = await ac.post(
            "/api/chat",
            data={"question": "这组图片有什么缺陷？", "session_id": "mm_multi_001"},
            files=files,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert data["mode"] == "multimodal"
    assert data["batch_size"] == 3
    assert len(data["multimodalResults"]) == 3
    for item in data["multimodalResults"]:
        assert item["answer"]
        assert "image_description" in item

    assert data.get("batchSummary"), "多图请求应返回 batchSummary 批量缺陷统计"

@pytest.mark.asyncio
async def test_chat_stream(mock_llm):


    async def fake_stream(messages, llm_getter, tool_map, io_executor,
                          max_tool_loop=5, llm_timeout=120):
        yield {"type": "token", "content": "这是流式"}
        yield {"type": "token", "content": "回答片段"}
        yield {"type": "done", "answer": "这是流式回答片段"}

    with patch("api_llm_pro_latest.stream_llm_with_tools", fake_stream):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/api/chat",
                data={"question": "你好", "session_id": "stream_sess_001", "stream": "true"},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "start"' in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body
    assert "这是流式回答片段" in body

@pytest.mark.asyncio
async def test_delete_session():


    import api_llm_pro_latest as mod

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.delete("/api/session/test_sess_del")

    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert data["message"] == "session deleted"
    assert data["session_id"] == "test_sess_del"

    mod.session_manager.delete.assert_called_once_with("test_sess_del")


@pytest.mark.asyncio
async def test_chat_multimodal_stream(mock_llm, mock_vision):


    import io
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/chat",
            data={"question": "这张图片有什么缺陷？",
                  "session_id": "mm_stream_001",
                  "stream": "true"},
            files={"images": ("test.png", io.BytesIO(png_bytes), "image/png")})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "start"' in body
    assert '"mode": "multimodal"' in body
    assert '"type": "image_done"' in body
    assert '"filename"' in body
    assert '"type": "done"' in body
    assert '"batch_size": 1' in body


@pytest.mark.asyncio
async def test_chat_multimodal_multi_stream(mock_llm, mock_vision):


    import io
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = [
            ("images", (f"test_{i}.png", io.BytesIO(png_bytes), "image/png"))
            for i in range(3)]
        response = await ac.post(
            "/api/chat",
            data={"question": "这组图片有什么缺陷？",
                  "session_id": "mm_multi_stream_001",
                  "stream": "true"},
            files=files)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "image_done"' in body

    assert body.count('"type": "image_done"') == 3
    assert '"type": "batch_summary"' in body
    assert '"type": "done"' in body
    assert '"batch_size": 3' in body
