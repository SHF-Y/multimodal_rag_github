"""vision_module 单元测试：图片编码与格式转换"""
import base64
import io
import pytest
from PIL import Image
from core.vision_module import image_to_base64
class TestImageToBase64:

    def test_returns_valid_jpeg_base64(self, tmp_path):
        """RGB 图片应编码为可解码的 JPEG base64"""
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        path = tmp_path / "test.png"
        img.save(path, format="PNG")
        b64 = image_to_base64(str(path))

        raw = base64.b64decode(b64)

        assert raw[:2] == b"\xff\xd8"

        decoded = Image.open(io.BytesIO(raw))
        assert decoded.mode == "RGB"

    def test_rgba_transparency_removed(self, tmp_path):
        """RGBA 图片（带透明通道）应转为 RGB，无 alpha"""
        img = Image.new("RGBA", (8, 8), (0, 0, 255, 128))
        path = tmp_path / "rgba.png"
        img.save(path, format="PNG")

        b64 = image_to_base64(str(path))
        raw = base64.b64decode(b64)
        decoded = Image.open(io.BytesIO(raw))
        assert decoded.mode == "RGB"

    def test_grayscale_converted_to_rgb(self, tmp_path):
        """灰度图也应转为 RGB，保证后端统一收到三通道图"""
        img = Image.new("L", (6, 6), 128)
        path = tmp_path / "gray.png"
        img.save(path, format="PNG")

        b64 = image_to_base64(str(path))
        raw = base64.b64decode(b64)
        decoded = Image.open(io.BytesIO(raw))
        assert decoded.mode == "RGB"

    def test_content_preserved(self, tmp_path):
        """编码-解码后像素内容应保持一致（尺寸不变）"""
        img = Image.new("RGB", (20, 30), (10, 20, 30))
        path = tmp_path / "c.png"
        img.save(path, format="PNG")

        b64 = image_to_base64(str(path))
        raw = base64.b64decode(b64)
        decoded = Image.open(io.BytesIO(raw))
        assert decoded.size == (20, 30)

    def test_missing_file_raises(self):
        """传入不存在的文件，应抛出 Error"""
        with pytest.raises(Exception):
            image_to_base64("definitely_not_exists.jpg")
