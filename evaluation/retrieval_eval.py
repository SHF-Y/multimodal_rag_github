"""
检索评估脚本

对每个测试问题调用系统的混合检索器（core.rag_module），
将检索返回的 top-k 文档文本与测试用例中的 golden_keywords（黄金关键词）做子串匹配，
计算检索质量指标：
  - HitRate@k（k=3/5）：top-k 结果中是否至少有一个文档命中任意关键词
  - MRR@k（k=3/5）：首个命中文档的排名倒数（1/排名），衡量"首个正确结果排得多靠前"
  - Precision@3：top-3 中命中关键词的文档占比

评估结果写入 retrieval_report.json（含 summary 汇总 + per_question 逐题明细）
"""
import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.rag_module import get_retriever

def load_test_cases(path: str) -> list:
    """读取测试用例 JSON 文件并解析为 Python 列表。

    参数:path (str): test_cases.json 的完整路径
    返回:list: 测试用例字典列表，每个字典至少包含 question、golden_keywords、category 字段
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_hit(doc_text: str, keywords: list) -> bool:
    """判断一段检索到的文档文本是否命中 golden_keywords 中的任意一个关键词。

    匹配规则：
      - 大小写不敏感
      - 只要任意一个非空关键词是 doc_text 的子串，即视为命中
      - 空关键词（""）被显式跳过：空串是任何文本的子串，不跳过会造成全部误判为命中
    """
    for kw in keywords:
        if kw and kw.lower() in doc_text.lower():
            return True
    return False

def compute_hit_and_mrr(texts: list, keywords: list, k: int):
    """在 top-k 结果上计算 HitRate@k 与 MRR@k。

    参数:
        texts (list): 检索返回的文档文本列表（按相关度从高到低排列）
        keywords (list): 黄金关键词列表
        k (int): 只考察前 k 个结果
    返回:
        (hit, rr) 元组：
          hit (bool): top-k 内是否存在至少一个命中文档
          rr (float): 第一个命中文档的 1/排名（排名从1开始），未命中则为 0.0

    实现说明：
      - 从第 1 个结果（索引0）开始逐个判断，找到第一个命中后立即 break
      - MRR 只由"第一个命中"的位置决定，后续是否命中不影响 rr
      - 找到第一个命中时 hit 即置为 True，因此 hit 等价于"top-k 内存在命中"
    """
    hit = False
    rr = 0.0
    for i, t in enumerate(texts[:k]):
        if is_hit(t, keywords):
            if not hit:
                rr = 1.0 / (i + 1)
            hit = True
            break
    return hit, rr

def main():

    cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    test_cases = load_test_cases(cases_path)
    n = len(test_cases)
    print(f"加载 {n} 个测试用例")

    print("\n初始化检索器（加载向量库 + 构建 BM25 索引）...")
    retriever = get_retriever(k=5)

    per_question = []
    for tc in test_cases:
        q = tc["question"]
        kws = tc.get("golden_keywords", [])
        try:

            docs = retriever.invoke(q)
            texts = [d.page_content for d in docs]
        except Exception as e:

            print(f"  [错误] {q[:30]}... {e}")
            per_question.append({"question": q, "error": str(e)})
            continue

        hit3, rr3 = compute_hit_and_mrr(texts, kws, 3)
        hit5, rr5 = compute_hit_and_mrr(texts, kws, 5)

        top3 = texts[:3]
        prec3 = sum(1 for t in top3 if is_hit(t, kws)) / len(top3) if top3 else 0.0

        per_question.append({
            "question": q,
            "category": tc.get("category", ""),
            "hit@3": hit3, "mrr@3": rr3,
            "hit@5": hit5, "mrr@5": rr5,
            "precision@3": prec3,
            "top_texts": [t[:120] for t in texts[:3]],
        })
        print(f"  [{tc.get('category','')}] {q[:28]}... hit@3={hit3} mrr@3={rr3:.3f}")

    ok = [r for r in per_question if "error" not in r]
    print(f"\n有效用例: {len(ok)}/{n}")

    def avg(key):
        vals = [r[key] for r in ok]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "总用例": n,
        "HitRate@3": avg("hit@3"),
        "MRR@3": avg("mrr@3"),
        "HitRate@5": avg("hit@5"),
        "MRR@5": avg("mrr@5"),
        "Precision@3": avg("precision@3"),
    }
    print("\n===== 检索评估汇总 =====")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_path = os.path.join(os.path.dirname(__file__), "retrieval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": ok},
                  f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")

if __name__ == "__main__":
    main()
