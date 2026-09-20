"""
evaluation 评测包

本包包含 RAG 系统的离线与在线评测工具，各文件职责如下：

  - rag_evaluator.py     RAGAS 在线评测器：调用 LLM 计算 Faithfulness / Answer Relevancy /
                         Context Precision / Context Recall / Answer Correctness 五项指标
  - run_evaluation.py    端到端评测入口：调用真实后端 /api/chat 获取回答后，用 RAGEvaluator 评分
  - retrieval_eval.py    离线检索评估：不调用生成式 LLM，只验证检索器命中率（HitRate / MRR / Precision）
  - vision_eval.py       视觉评估：调用多模态接口对缺陷图片分类，计算分类准确率与混淆矩阵
  - dataset_builder.py   评测数据集构建器：从知识库 PDF 生成问答对，供人工校验后作为测试用例
  - test_cases.json      检索/端到端评测的测试用例数据（question / ground_truth / golden_keywords）
  - retrieval_report.json retrieval_eval.py 生成的检索评估结果报告（自动生成，可被覆盖）
  - vision_report.json    vision_eval.py 生成的视觉评估结果报告（首次运行后生成）
"""
