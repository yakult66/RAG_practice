import argparse
import json

from Embedding import DOCS_DIR, Embedding


def is_match(hit, case):
    if hit["meta"].get("name") != case["expected_source"]:
        return False
    return not case.get("expected_text") or case["expected_text"] in hit["document"]


def evaluate(rag, cases, k):
    found, reciprocal_ranks, misses = 0, [], []
    for case in cases:
        hits = rag.retrieve(case["question"], top_k=k, max_distance=2.0)
        rank = next((i for i, h in enumerate(hits, 1) if is_match(h, case)), None)
        if rank:
            found += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append((case["question"], [Embedding.source_label(h["meta"]) for h in hits]))
        best = f"{hits[0]['distance']:.3f}" if hits else "-"
        print(f"{'✓' if rank else '✗'} 名次 {rank or '-'}｜最近距離 {best}｜{case['question']}")

    n = len(cases)
    print(f"\nRecall@{k}: {found / n:.2%}（{found}/{n}）")
    print(f"MRR@{k}: {sum(reciprocal_ranks) / n:.3f}")
    for q, labels in misses:
        print(f"\n沒找到：{q}\n  實際取回：{labels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="評估檢索品質（只呼叫 embedding API，不呼叫 LLM）")
    parser.add_argument("--set", default="eval/eval_set.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-index", action="store_true", help="略過建索引，直接評估")
    args = parser.parse_args()

    rag = Embedding()
    if not args.no_index:
        rag.index_folder(DOCS_DIR)
    with open(args.set, encoding="utf-8") as f:
        evaluate(rag, json.load(f), args.k)
