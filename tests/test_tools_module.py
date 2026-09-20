
"""
被测模块给 LLM提供了三个"工具"：
      1. extract_image_text        —— 提取图片里的文字
      2. defect_type_identify      —— 识别工业零件图片的缺陷类型
      3. batch_defect_statistics   —— 批量统计一个文件夹里所有图片的缺陷
    单元测试要验证的是"工具是否用正确的提示词、正确的流程调用了视觉模型"。
"""

from unittest.mock import MagicMock, patch

import os

from core.tools_module import (
    ALL_TOOLS,
    batch_defect_statistics,
    defect_type_identify,
    extract_image_text,
)

class TestExtractImageText:
    """extract_image_text 函数测试
    验证提示词是否要求"精准提取、按原文输出"。
    """
    def test_returns_extracted_text(self):
        """验证：提示词正确 + 返回格式正确。"""
        with patch("core.tools_module.parse_image", return_value="锈蚀缺陷：轻中度") as mock_parse:
            result = extract_image_text.invoke({"img_path": "/fake/img.png"})
        mock_parse.assert_called_once_with(
            "/fake/img.png", "精准提取图片中所有文字，按原文输出，无额外说明"
        )

        assert "图片文字提取结果" in result
        assert "锈蚀缺陷：轻中度" in result

class TestDefectTypeIdentify:
    """defect_type_identify 函数测试
    验证提示词约束输出格式。
    """

    def test_returns_defect_type(self):
        """验证：提示词正确 + 返回格式正确。"""
        with patch("core.tools_module.parse_image", return_value="锈蚀，中度") as mock_parse:
            result = defect_type_identify.invoke({"img_path": "/fake/part.png"})

        mock_parse.assert_called_once_with(
            "/fake/part.png",
            "判断该设备缺陷类型，仅返回缺陷名称和严重程度，无缺陷返回'合格'",
        )
        assert "缺陷识别结果" in result
        assert "锈蚀，中度" in result

class TestBatchDefectStatistics:
    """batch_defect_statistics 函数测试
       验证：  空目录/无图片时返回提示语；
         扩展名过滤正确；
         同一缺陷多次出现时计数累加；
    """

    def test_returns_message_when_no_images(self):
        """文件夹里没有任何图片 → 返回"未找到"提示，且不调用模型。"""

        with patch("core.tools_module.os.listdir", return_value=["readme.txt", "data.csv"]), \
             patch("core.tools_module.parse_image") as mock_parse:
            result = batch_defect_statistics.invoke({"folder_path": "/fake/folder"})

        assert "未找到有效图片文件" in result

        mock_parse.assert_not_called()

    def test_counts_defects_by_type(self):
        """正常流程：
        多张图片按缺陷类型分组计数；
        非图片文件被过滤；
        扩展名大小写不敏感。"""
        with patch("core.tools_module.os.listdir",return_value=["a.jpg", "b.JPG", "c.png", "notes.txt"]), \
             patch("core.tools_module.parse_image", side_effect=["锈蚀", "合格", "渗漏油"]) as mock_parse:
            
            result = batch_defect_statistics.invoke({"folder_path": "/fake/folder"})

        assert "共扫描 3 张图片" in result

        assert "锈蚀: 1 件" in result
        assert "渗漏油: 1 件" in result
        assert "合格: 1 件" in result

        assert mock_parse.call_count == 3

        mock_parse.assert_any_call(os.path.join("/fake/folder", "a.jpg"), "判断这张图片是否包含缺陷，仅返回缺陷类型名称，比如“锈蚀”、“渗漏油”，无缺陷则返回'合格'，不要多余内容")

    def test_same_defect_accumulates(self):
        """同一种缺陷出现多次 → 计数必须累加，而不是互相覆盖。"""

        with patch("core.tools_module.os.listdir", return_value=["1.jpg", "2.jpg", "3.jpg"]), \
             patch("core.tools_module.parse_image",  side_effect=["锈蚀", "锈蚀", "渗漏油"]):
            result = batch_defect_statistics.invoke({"folder_path": "/fake/folder"})

        assert "锈蚀: 2 件" in result
        assert "渗漏油: 1 件" in result

class TestAllTools:
    """ALL_TOOLS 测试"""

    def test_registers_all_tools(self):
        """工具清单必须包含且仅包含三个工具，且名称与函数一一对应。
        StructuredTool 对象有 .name 属性（就是被装饰的函数名）。"""
        names = {t.name for t in ALL_TOOLS}
        assert len(ALL_TOOLS) == 3
        assert names == {"extract_image_text", "defect_type_identify", "batch_defect_statistics"}
