"""
评测数据集构建器
从知识库PDF中自动生成问答对，再人工校验

功能概述：
1. 加载指定文件夹下的所有PDF文件
2. 将每个PDF按规则切分成文本块
3. 调用大语言模型（LLM）为每个文本块生成3-5个问答对
4. 汇总所有问答对，去重后保存为JSON文件
5. 对生成的问题进行简单分类统计
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)

import json
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.messages import SystemMessage, HumanMessage

from core.config import get_llm

GENERATE_QA_SYSTEM = """你是一个专业的数据集构建专家。请基于给定的文本片段，生成3-5个高质量的问答对。

要求：
1. 问题应该多样化：包括事实查询、概念解释、标准数值、操作流程等类型
2. 答案必须完全基于给定文本，不能编造
3. 问题应该是用户真实会问的形式，不要太学术化
4. 每个问答对包含：question（问题）、answer（标准答案）、source（来源文本片段）
5. 返回JSON格式：{"qa_pairs": [{"question": "...", "answer": "...", "source": "..."}]}
"""

class DatasetBuilder:
    """
    数据集构建器类
    负责从PDF文档中提取文本块，调用LLM生成问答对，并整理成数据集
    """

    def __init__(self):
        """
        初始化构建器
        从配置中获取LLM实例，后续生成问答对时使用
        """
        self.llm = get_llm()

    def load_pdf_chunks(self, pdf_path: str, chunk_size: int = 800) -> List[str]:
        """
        加载指定PDF文件并将其切分成文本块
        参数:
            pdf_path (str): PDF文件的路径
            chunk_size (int): 每个文本块的最大字符数，默认800
        返回:
            List[str]: 切分后的文本块列表，每个元素是一个字符串
        """

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,

            separators=["\n\n", "\n", "。", "；", "，", " ", ""]
        )

        chunks = splitter.split_documents(docs)

        return [doc.page_content for doc in chunks]

    def generate_qa_from_chunk(self, chunk: str) -> List[Dict]:
        """
        调用LLM，基于给定的文本块生成问答对列表
        参数:
            chunk (str): 一个文本块的内容
        返回:
            List[Dict]: 问答对字典列表，每个字典包含question、answer、source字段
                        如果解析失败，返回空列表
        """

        messages = [

            SystemMessage(content=GENERATE_QA_SYSTEM),

            HumanMessage(content=f"文本片段：\n{chunk}\n\n请生成问答对：")
        ]

        result = self.llm.invoke(messages)

        try:

            content = result.content.strip()

            if content.startswith("```"):

                content = content.split("\n", 1)[1]

                if content.endswith("```"):

                    content = content.rsplit("```", 1)[0]
                    

            data = json.loads(content)

            return data.get("qa_pairs", [])
        except Exception as e:

            print(f"解析问答对失败: {e}")
            return []

    def build_dataset(self, pdf_folder: str, output_path: str,
                      max_chunks: int = 50):
        """
        从指定文件夹中的所有PDF文件构建完整评测数据集，并保存到JSON文件

        参数:
            pdf_folder (str): 存放PDF文件的文件夹路径
            output_path (str): 输出JSON文件的保存路径
            max_chunks (int): 每个PDF最多处理的文本块数量，避免处理过长文档导致成本过高，默认50
        返回:
            List[Dict]: 生成的所有问答对列表（去重后）
        """

        all_qa = []

        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

        for pdf_file in pdf_files:

            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"处理PDF: {pdf_file}")

            chunks = self.load_pdf_chunks(pdf_path)

            chunks = chunks[:max_chunks]

            for i, chunk in enumerate(chunks):

                print(f"  生成第 {i+1}/{len(chunks)} 块的问答对...")

                qa_pairs = self.generate_qa_from_chunk(chunk)

                for qa in qa_pairs:
                    qa["source_file"] = pdf_file
                    all_qa.append(qa)

        seen_questions = set()
        unique_qa = []

        for qa in all_qa:

            q = qa["question"].strip()

            if q not in seen_questions:
                seen_questions.add(q)
                unique_qa.append(qa)

        with open(output_path, "w", encoding="utf-8") as f:

            json.dump(unique_qa, f, ensure_ascii=False, indent=2)

        print(f"\n数据集构建完成！共 {len(unique_qa)} 个问答对，保存到: {output_path}")

        return unique_qa

    def classify_questions(self, qa_pairs: List[Dict]) -> Dict:
        """
        对生成的问题进行简单分类统计（基于关键词匹配）

        参数:
            qa_pairs (List[Dict]): 问答对列表（每个字典需含 question 字段）
        返回:
            Dict: 分类结果字典，键为类别名称，值为属于该类别的问答对列表。
                  类别包括：事实查询、标准数值、操作流程、原因分析、判断鉴别、其他。
        """

        categories = {
            "事实查询": ["什么是", "定义", "概念", "含义"],
            "标准数值": ["多少", "标准", "限值", "要求", "参数"],
            "操作流程": ["如何", "怎么", "步骤", "流程", "方法"],
            "原因分析": ["为什么", "原因", "导致", "影响"],
            "判断鉴别": ["判断", "区别", "对比", "鉴别", "识别"]
        }

        result = {cat: [] for cat in categories}
        result["其他"] = []

        for qa in qa_pairs:
            question = qa["question"]
            classified = False

            for cat, keywords in categories.items():

                if any(kw in question for kw in keywords):
                    result[cat].append(qa)
                    classified = True
                    break

            if not classified:
                result["其他"].append(qa)

        return result

if __name__ == "__main__":

    builder = DatasetBuilder()

    dataset = builder.build_dataset(
        pdf_folder="./data/pdfs",
        output_path="./evaluation/industrial_qa_dataset.json",
        max_chunks=5
    )

    stats = builder.classify_questions(dataset)

    print("\n问题分类统计：")
    for cat, items in stats.items():
        print(f"  {cat}: {len(items)} 题")

