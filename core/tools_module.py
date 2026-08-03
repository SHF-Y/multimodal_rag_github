from langchain_core.tools import tool
from .vision_module import parse_image
import os

@tool
def extract_image_text(img_path: str) -> str:
    """提取图片中的所有文字内容，输入为图片本地路径"""
    result = parse_image(img_path, "精准提取图片中所有文字，按原文输出，无额外说明")
    return f"图片文字提取结果：\n{result}"

@tool
def defect_type_identify(img_path: str) -> str:
    """识别工业零件图片的缺陷类型，输入为图片本地路径"""
    result = parse_image(img_path, "判断该零件缺陷类型，仅返回缺陷名称和严重程度，无缺陷返回'合格'")
    return f"缺陷识别结果：{result}"

# ===================== 批量统计工具 =====================
@tool
def batch_defect_statistics(folder_path: str) -> str:#定义工具的执行逻辑。
    
    """
    批量统计指定文件夹内所有图片的缺陷类型与数量。
    当用户需要统计某个文件夹下的缺陷时，调用此工具。
    返回一个包含每张图片缺陷统计的 JSON 字符串。
    输入参数：文件夹的绝对路径,（如 f:/data/images）
    返回：缺陷类型统计结果报表。
    """
    image_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_ext]
    if not image_files:
        return "文件夹内未找到有效图片文件"
    
    result = f"共扫描 {len(image_files)} 张图片：\n"
    defect_count = {}
    
    for img_file in image_files:
        img_full_path = os.path.join(folder_path, img_file)#
        desc = parse_image(img_full_path, "判断这张图片是否包含缺陷，仅返回缺陷类型名称，比如“锈蚀”或“渗漏油”，无缺陷则返回'合格'，不要多余内容")
        defect_type = desc.strip()
        defect_count[defect_type] = defect_count.get(defect_type, 0) + 1
        
    
    result += "缺陷统计结果：\n"
    for k, v in defect_count.items():
        result += f"- {k}: {v} 件\n"
    
    return result
# 工具列表
ALL_TOOLS = [extract_image_text, defect_type_identify, batch_defect_statistics]