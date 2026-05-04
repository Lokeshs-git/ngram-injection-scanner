"""Zero-shot scanner evaluation on public prompt injection benchmarks.

Evaluates the scanner on three publicly available datasets for comparability
with prior work. No training, no fine-tuning — purely zero-shot.

Datasets:
  - MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT (gated, requires HF_TOKEN)
      1000-sample stratified held-out set (500 injections / 500 clean, seed=42)
      Context field = external retrieved document, same framing as our main eval.
      Supervised baseline: Alamsabi et al. (MDPI Algorithms 2026) F1=0.977
      using XGBoost + OpenAI embeddings trained on this dataset.

  - deepset/prompt-injections (test split: 116 samples, 60 injections)
  - xTRam1/safe-guard-prompt-injection (test split: 2060 samples, 650 injections)
      Supervised baseline: Ayub & Majumdar (arXiv:2410.22284) AUC=0.764, F1=0.867

Output: results/public_benchmarks.json

Usage:
    uv run python scripts/eval_public_benchmarks.py
"""

import json
import os
import random
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, roc_auc_score

load_dotenv(".env")

REFERENCE_PATH = "data/reference_set.json"
OUT_PATH = "results/public_benchmarks.json"

SCANNER_MODEL = "all-MiniLM-L6-v2"
NGRAM_N = 5
BATCH_SIZE = 32
SEED = 42

# BIPIA held-out sample size (stratified, no official test split)
BIPIA_SAMPLE_PER_CLASS = 500

DATASETS: dict[str, dict] = {
    "bipia": {
        "id": "MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT",
        "split": "train",
        "text_col": "context",
        "label_col": "label",
        "gated": True,
        "sample_per_class": BIPIA_SAMPLE_PER_CLASS,
        "supervised_baseline": "Alamsabi et al. (MDPI 2026): F1=0.977, XGBoost + OpenAI emb (trained on this data)",
    },
    "deepset": {
        "id": "deepset/prompt-injections",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "gated": False,
        "supervised_baseline": "Ayub & Majumdar (arXiv:2410.22284): AUC=0.764, F1=0.867, RF + OpenAI emb",
    },
    "xTRam1": {
        "id": "xTRam1/safe-guard-prompt-injection",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "gated": False,
        "supervised_baseline": "Ayub & Majumdar (arXiv:2410.22284): AUC=0.764, F1=0.867, RF + OpenAI emb",
    },
}


def stratified_sample(
    texts: list[str],
    labels: list[int],
    n_per_class: int,
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    pos = [(t, lbl) for t, lbl in zip(texts, labels, strict=False) if lbl == 1]
    neg = [(t, lbl) for t, lbl in zip(texts, labels, strict=False) if lbl == 0]
    sampled = rng.sample(pos, min(n_per_class, len(pos))) + rng.sample(
        neg, min(n_per_class, len(neg))
    )
    rng.shuffle(sampled)
    return [s[0] for s in sampled], [s[1] for s in sampled]


def run_scanners(
    texts: list[str],
    model: SentenceTransformer,
    ref_embs: torch.Tensor,
    n: int = NGRAM_N,
) -> dict[str, list[float]]:
    global_scores: list[float] = []
    token_scores: list[float] = []
    ngram_scores: list[float] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = model.tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        mask = inputs["attention_mask"]
        with torch.no_grad():
            te = model[0](inputs)["token_embeddings"]

        g = F.normalize(torch.mean(te * mask.unsqueeze(-1), dim=1), p=2, dim=1)
        global_scores.extend(torch.max(torch.matmul(g, ref_embs.T), dim=1).values.cpu().numpy())

        nt = F.normalize(te, p=2, dim=2)
        st, _ = torch.max(torch.matmul(nt, ref_embs.T), dim=2)
        st = st * mask - (1 - mask) * 999.0
        token_scores.extend(torch.max(st, dim=1).values.cpu().numpy())

        if te.shape[1] >= n:
            ng = F.avg_pool1d(te.transpose(1, 2), kernel_size=n, stride=1).transpose(1, 2)
            ng = F.normalize(ng, p=2, dim=2)
            sng, _ = torch.max(torch.matmul(ng, ref_embs.T), dim=2)
            ng_mask = mask[:, : sng.shape[1]]
            sng = sng * ng_mask - (1 - ng_mask) * 999.0
            ngram_scores.extend(torch.max(sng, dim=1).values.cpu().numpy())
        else:
            ngram_scores.extend(torch.max(st, dim=1).values.cpu().numpy())

    return {
        "global": [float(x) for x in global_scores],
        "token_level": [float(x) for x in token_scores],
        f"ngram_n{n}": [float(x) for x in ngram_scores],
    }


def best_f1(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Return best F1 and the threshold that achieves it."""
    thresholds = sorted(set(scores))
    best, best_t = 0.0, 0.0
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        f = f1_score(labels, preds, zero_division=0)
        if f > best:
            best, best_t = f, t
    return round(float(best), 4), round(float(best_t), 6)


if __name__ == "__main__":
    rng = random.Random(SEED)
    hf_token = os.environ.get("HF_TOKEN")

    print("Loading reference set...")
    with open(REFERENCE_PATH) as f:
        ref_data = json.load(f)
    reference_phrases = [r["goal"] for r in ref_data["reference_set"]]

    print(f"Loading scanner model ({SCANNER_MODEL})...")
    scanner = SentenceTransformer(SCANNER_MODEL)
    ref_embs = F.normalize(scanner.encode(reference_phrases, convert_to_tensor=True), p=2, dim=1)

    results: dict = {
        "scanner_model": SCANNER_MODEL,
        "n_reference_phrases": len(reference_phrases),
        "ngram_n": NGRAM_N,
        "evaluation": "zero-shot — no training on any injection dataset",
        "datasets": {},
    }

    for ds_key, cfg in DATASETS.items():
        if cfg.get("gated") and not hf_token:
            print(f"\nSkipping {ds_key} — gated dataset, HF_TOKEN not set.")
            continue

        print(f"\nLoading {cfg['id']} ({cfg['split']})...")
        load_kwargs: dict = {"split": cfg["split"]}
        if cfg.get("gated"):
            load_kwargs["token"] = hf_token

        ds = load_dataset(cfg["id"], **load_kwargs)
        texts = [str(r[cfg["text_col"]]) for r in ds]
        labels = [int(r[cfg["label_col"]]) for r in ds]

        # Stratified sample for datasets without an official test split
        if "sample_per_class" in cfg:
            texts, labels = stratified_sample(texts, labels, cfg["sample_per_class"], rng)
            print(
                f"  Sampled {len(texts)} ({sum(labels)} injections / {len(labels) - sum(labels)} clean, seed={SEED})"
            )
        else:
            print(
                f"  {len(texts)} samples, {sum(labels)} injections ({100 * sum(labels) / len(texts):.1f}%)"
            )

        t0 = time.perf_counter()
        scores = run_scanners(texts, scanner, ref_embs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        dataset_results: dict = {
            "n_samples": len(texts),
            "n_injections": sum(labels),
            "supervised_baseline": cfg.get("supervised_baseline", ""),
            "latency_per_sample_ms": round(elapsed_ms / len(texts), 2),
            "methods": {},
        }

        print(f"  {'Method':<15} {'AUC':>8} {'Best F1':>9}")
        print(f"  {'-' * 34}")
        for method, method_scores in scores.items():
            auc = round(float(roc_auc_score(labels, method_scores)), 4)
            f1, thresh = best_f1(method_scores, labels)
            dataset_results["methods"][method] = {
                "auc": auc,
                "best_f1": f1,
                "best_f1_threshold": thresh,
            }
            marker = " *" if method == f"ngram_n{NGRAM_N}" else ""
            print(f"  {method:<15} {auc:>8.4f} {f1:>9.4f}{marker}")

        results["datasets"][ds_key] = dataset_results

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 65}")
    print("SUMMARY — zero-shot vs. supervised baselines")
    print(f"{'=' * 65}")
    for ds_key, dr in results["datasets"].items():
        m = dr["methods"].get(f"ngram_n{NGRAM_N}", {})
        print(f"\n  {ds_key}")
        print(f"    Supervised baseline: {dr['supervised_baseline']}")
        print(f"    Ours (zero-shot N-gram N=5): AUC={m.get('auc')}  F1={m.get('best_f1')}")

    print(f"\nSaved to {OUT_PATH}")
