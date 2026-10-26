"""Usage: python eval.py [predictions.json] [gold.json]"""
import json
import sys
from collections import Counter

VERDICTS = ["supported", "overstated", "cannot_verify"]


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("predictions") or data.get("gold") or data.get("claims") or []
    return {item["claim_id"]: item for item in data}


def main():
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/predictions.json"
    gold_path = sys.argv[2] if len(sys.argv) > 2 else "data/gold.json"
    preds, gold = load(pred_path), load(gold_path)

    matrix, correct, false_supported = Counter(), 0, 0
    for cid, g in gold.items():
        p = preds.get(cid, {}).get("verdict", "missing")
        matrix[(g["verdict"], p)] += 1
        correct += p == g["verdict"]
        false_supported += p == "supported" and g["verdict"] != "supported"
        if p != g["verdict"]:
            print(f"  {cid}: gold={g['verdict']} predicted={p}")

    total = len(gold)
    print(f"\nAccuracy: {correct}/{total} = {correct / total:.1%}" if total else "No gold claims.")
    cols = VERDICTS + ["missing"]
    print("\nConfusion matrix (rows = gold, cols = predicted)")
    print(f"{'':16}" + "".join(f"{c:>15}" for c in cols))
    for r in VERDICTS:
        print(f"{r:16}" + "".join(f"{matrix[(r, c)]:>15}" for c in cols))
    print(f"\nFALSE 'supported' verdicts: {false_supported}")


if __name__ == "__main__":
    main()
