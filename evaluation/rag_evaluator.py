"""RAG系统自动化评测
使用RAGAS框架评测以下指标：
1. Faithfulness（忠实度）：回答是否基于检索到的上下文，有无幻觉
2. Answer Relevancy（回答相关性）：回答是否切题
3. Context Precision（上下文精确率）：检索到的上下文是否相关
4. Context Recall（上下文召回率）：相关上下文是否被检索到
5. Answer Correctness（回答正确率）：回答与标准答案的匹配度
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

from core.config import get_llm, get_embedding
class RAGEvaluator:
    """
    RAG 系统评估器，封装了数据集构建、评估执行和报告生成的功能。
    """
    def __init__(self):
        """
        初始化评估器，加载裁判模型 LLM 与向量模型 Embeddings。
        """

        self.llm = get_llm(temperature=0)

        self.embeddings = get_embedding()

    def build_eval_dataset(self, test_cases: List[Dict]) -> Dataset:
        """
        将测试用例列表转换为 RAGAS 所需的 Hugging Face Dataset 对象。
        test_cases: 测试用例列表，每个元素是一个字典，必须包含以下键：
            - "question": 用户提出的问题（字符串）
            - "answer": RAG 系统生成的回答（字符串）
            - "contexts": 检索到的上下文列表（字符串列表）
            - "ground_truth": 标准答案（字符串）
        返回Dataset: 包含（question, answer, contexts, ground_truth）的数据集，直接传递给 ragas.evaluate 函数。
        """

        data = {
            "question": [tc["question"] for tc in test_cases],
            "answer": [tc["answer"] for tc in test_cases],
            "contexts": [tc["contexts"] for tc in test_cases],
            "ground_truth": [tc["ground_truth"] for tc in test_cases]
        }

        return Dataset.from_dict(data)

    def evaluate(self, test_cases: List[Dict]) -> Dict:
        """执行完整的 RAG 评估流程。
        参数test_cases: 
        返回Dict, 包含两大主要部分：
                - 每个指标的逐题得分列表（键名与指标名相同）
                - "overall" 键:每个指标的平均分数（算术平均）
        """

        self._test_questions = [tc["question"] for tc in test_cases]

        dataset = self.build_eval_dataset(test_cases)

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,answer_relevancy,
                context_precision,context_recall,
                answer_correctness],
            llm=self.llm,
            embeddings=self.embeddings
        )

        result_dict = result.to_pandas().to_dict(orient="list")

        return {
            "faithfulness": result_dict.get("faithfulness", []),
            "answer_relevancy": result_dict.get("answer_relevancy", []),
            "context_precision": result_dict.get("context_precision", []),
            "context_recall": result_dict.get("context_recall", []),
            "answer_correctness": result_dict.get("answer_correctness", []),
            "overall": {
                "faithfulness": sum(result_dict.get("faithfulness", [])) / len(result_dict.get("faithfulness", [1])),
                "answer_relevancy": sum(result_dict.get("answer_relevancy", [])) / len(result_dict.get("answer_relevancy", [1])),
                "context_precision": sum(result_dict.get("context_precision", [])) / len(result_dict.get("context_precision", [1])),
                "context_recall": sum(result_dict.get("context_recall", [])) / len(result_dict.get("context_recall", [1])),
                "answer_correctness": sum(result_dict.get("answer_correctness", [])) / len(result_dict.get("answer_correctness", [1]))
            }
        }

    def generate_report(self, results: Dict, output_path: str):
        """
        根据前面 evaluate 的结果生成 Markdown 格式的评测报告并保存到文件。
        参数results: evaluate 方法返回的字典，包含逐题分数和总体平均分。
            output_path: 报告文件的保存路径（例如 "report.md"）。
        """

        overall = results["overall"]

        report = f"""# RAG系统评测报告

## 总体指标

| 指标 | 分数 | 说明 |
|------|------|------|
| Faithfulness（忠实度） | {overall['faithfulness']:.4f} | 回答是否基于上下文，有无幻觉 |
| Answer Relevancy（回答相关性） | {overall['answer_relevancy']:.4f} | 回答是否切题 |
| Context Precision（上下文精确率） | {overall['context_precision']:.4f} | 检索上下文的相关比例 |
| Context Recall（上下文召回率） | {overall['context_recall']:.4f} | 相关上下文被检索到的比例 |
| Answer Correctness（回答正确率） | {overall['answer_correctness']:.4f} | 与标准答案的匹配度 |

## 逐题详情

| 题号 | 问题 | 忠实度 | 相关性 | 精确率 | 召回率 | 正确率 |
|------|------|--------|--------|--------|--------|--------|
"""

        for i in range(len(results["faithfulness"])):

            question = self._test_questions[i] if hasattr(self, '_test_questions') else "..."

            question_display = question[:30] + "..." if len(question) > 30 else question

            report += (f"| {i+1} | {question_display} | "
                      f"{results['faithfulness'][i]:.4f} | "
                      f"{results['answer_relevancy'][i]:.4f} | "
                      f"{results['context_precision'][i]:.4f} | "
                      f"{results['context_recall'][i]:.4f} | "
                      f"{results['answer_correctness'][i]:.4f} |\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"评测报告已生成: {output_path}")