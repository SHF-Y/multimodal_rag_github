"""
  1. 加载测试用例
  2. 对每个问题调用本地后端网关 /api/chat，获取系统回答与检索上下文
  3. 将回答、上下文、标准答案交给 RAGEvaluator，用 RAGAS 框架计算 5 项指标
     （Faithfulness / Answer Relevancy / Context Precision / Context Recall / Answer Correctness）
  4. 生成 Markdown 评测报告 evaluation_report.md

"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

import json
import requests
import time

from evaluation.rag_evaluator import RAGEvaluator

def run_system_query(question: str) -> dict:
    """调用统一对话网关 /api/chat（非流式），获取回答与引用文档。

    参数:   question (str): 用户问题
    返回:dict: {"answer": 系统回答文本, "contexts": 检索引用文档列表}
              若 HTTP 状态码不是 200，返回 {"answer": "调用失败", "contexts": []}
    """
    resp = requests.post(
        "http://127.0.0.1:8000/api/chat",
        files=[
            ("question", (None, question)),
            ("session_id", (None, f"eval_{int(time.time() * 1000)}")),
        ],
        timeout=60
    )
    if resp.status_code == 200:
        data = resp.json()
        return {
            "answer": data["answer"],
            "contexts": data.get("docs", [])
        }
    return {"answer": "调用失败", "contexts": []}

def main():

    with open(os.path.join(EVAL_DIR, "test_cases1.json"), "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"加载了 {len(test_cases)} 个测试用例")

    for i, tc in enumerate(test_cases):
        print(f"处理第 {i+1}/{len(test_cases)} 题: {tc['question'][:30]}...")
        result = run_system_query(tc["question"])
        tc["answer"] = result["answer"]

        tc["contexts"] = result["contexts"]

    evaluator = RAGEvaluator()
    results = evaluator.evaluate(test_cases)

    evaluator.generate_report(results, os.path.join(EVAL_DIR, "evaluation_report.md"))

    print("\n===== 评测结果 =====")
    for metric, score in results["overall"].items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
