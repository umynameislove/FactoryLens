"""Score the full hazelnut test set and check separation / threshold quality.

Answers three questions before the demo:
  1. How many `good` images are wrongly flagged at the current threshold?
  2. How well does the score separate good vs defect at all (AUROC)?
  3. What threshold would actually be optimal for this memory bank?

Usage
-----
    python scripts/eval_threshold_full.py
    python scripts/eval_threshold_full.py --threshold 0.3133 --limit 5   # quick smoke

Writes `assets/eval_samples/eval_scores_full.csv`
(columns: image_path,label,is_defect,score) which is compatible with
`scripts/eval_per_type.py`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factorylens.vision.anomaly import load_memory_bank, score_image  # noqa: E402
from factorylens.vision.embeddings import PatchEmbeddingExtractor  # noqa: E402

DEFECT_LABELS = ("crack", "cut", "hole", "print")
TEST_ROOT = PROJECT_ROOT / "data/mvtec/hazelnut/test"
DEFAULT_BANK = PROJECT_ROOT / "data/memory_bank.npz"
DEFAULT_OUT = PROJECT_ROOT / "assets/eval_samples/eval_scores_full.csv"


def collect_images(limit: int | None) -> list[tuple[Path, str, bool]]:
    items: list[tuple[Path, str, bool]] = []
    for label in ("good",) + DEFECT_LABELS:
        folder = TEST_ROOT / label
        if not folder.is_dir():
            print(f"  ! missing folder: {folder}")
            continue
        paths = sorted(folder.glob("*.png"))
        if limit:
            paths = paths[:limit]
        for p in paths:
            items.append((p, label, label != "good"))
    return items


def auroc(scores: np.ndarray, is_defect: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties handled with average ranks."""
    pos = scores[is_defect]
    neg = scores[~is_defect]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_sum_pos = ranks[is_defect].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics_at(scores: np.ndarray, is_defect: np.ndarray, thr: float) -> dict:
    pred = scores >= thr
    tp = int((pred & is_defect).sum())
    fp = int((pred & ~is_defect).sum())
    fn = int((~pred & is_defect).sum())
    tn = int((~pred & ~is_defect).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(scores)
    tpr = rec
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "youden": tpr - fpr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3133,
                    help="current production threshold to evaluate")
    ap.add_argument("--memory-bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="only N images per class (quick smoke test)")
    ap.add_argument("--image-size", type=int, default=None,
                    help="override extractor input resize (must match how the "
                         "memory bank was built; v2 bundle used 512)")
    args = ap.parse_args()

    if not args.memory_bank.is_file():
        raise SystemExit(f"memory bank not found: {args.memory_bank}")

    items = collect_images(args.limit)
    if not items:
        raise SystemExit(f"no test images under {TEST_ROOT}")
    print(f"Scoring {len(items)} images with {args.memory_bank.name} ...")

    bank, dist_scale = load_memory_bank(str(args.memory_bank))
    print(f"  memory_bank shape={bank.shape}  distance_scale={dist_scale}")
    if args.image_size:
        extractor = PatchEmbeddingExtractor(image_size=args.image_size)
        print(f"  extractor image_size={args.image_size}")
    else:
        extractor = PatchEmbeddingExtractor()
        print(f"  extractor image_size={extractor.image_size} (repo default)")

    rows = []
    for i, (path, label, defect) in enumerate(items, 1):
        score, _ = score_image(
            str(path), memory_bank=bank, extractor=extractor,
            distance_scale=dist_scale,
        )
        rows.append({
            "image_path": str(path.relative_to(PROJECT_ROOT)),
            "label": label, "is_defect": int(defect), "score": round(score, 6),
        })
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_path", "label", "is_defect", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved scores -> {args.out.relative_to(PROJECT_ROOT)}")

    scores = np.array([r["score"] for r in rows], dtype=float)
    is_defect = np.array([bool(r["is_defect"]) for r in rows])
    labels = [r["label"] for r in rows]

    print("\n=== Score distribution per class ===")
    print(f"{'class':8} {'n':>4} {'min':>8} {'mean':>8} {'median':>8} {'max':>8}")
    for label in ("good",) + DEFECT_LABELS:
        sel = np.array([lb == label for lb in labels])
        if not sel.any():
            continue
        s = scores[sel]
        print(f"{label:8} {len(s):>4} {s.min():>8.4f} {s.mean():>8.4f} "
              f"{np.median(s):>8.4f} {s.max():>8.4f}")

    good = scores[~is_defect]
    bad = scores[is_defect]
    print("\n=== Separation ===")
    print(f"AUROC                 : {auroc(scores, is_defect):.4f}   (1.0 perfect, 0.5 random)")
    print(f"good  max             : {good.max():.4f}")
    print(f"defect min            : {bad.min():.4f}")
    overlap = good.max() >= bad.min()
    print(f"overlap good/defect   : {'YES - no threshold separates them cleanly' if overlap else 'no'}")

    m = metrics_at(scores, is_defect, args.threshold)
    print(f"\n=== At current threshold {args.threshold} ===")
    print(f"  false positives (good flagged) : {m['fp']} / {len(good)}")
    print(f"  false negatives (defect missed): {m['fn']} / {len(bad)}")
    print(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
          f"f1={m['f1']:.3f}  accuracy={m['accuracy']:.3f}")

    cands = np.unique(np.round(scores, 6))
    best_f1 = max((metrics_at(scores, is_defect, t) for t in cands), key=lambda d: d["f1"])
    best_j = max((metrics_at(scores, is_defect, t) for t in cands), key=lambda d: d["youden"])
    print("\n=== Best achievable thresholds on this set ===")
    print(f"  best F1     : thr={best_f1['threshold']:.4f}  f1={best_f1['f1']:.3f}  "
          f"fp={best_f1['fp']} fn={best_f1['fn']}")
    print(f"  best Youden : thr={best_j['threshold']:.4f}  J={best_j['youden']:.3f}  "
          f"fp={best_j['fp']} fn={best_j['fn']}")

    print("\nNOTE: threshold tuned on this same test set is optimistic (no held-out "
          "split). Report it as a calibration check, not as a validated metric.")


if __name__ == "__main__":
    main()
