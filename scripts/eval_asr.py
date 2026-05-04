"""ASR threshold analysis: scanner operating points across escalation rates.

Loads scanner raw_scores and labels from main_evaluation.json.
For each domain, sweeps flagging thresholds and reports:
  - Escalation rate: fraction of documents sent to the judge
  - TPR: fraction of poisoned documents caught (= 1 - ASR_proxy)
  - ASR proxy: fraction of poisoned documents missed (potential attack success)
  - FPR: fraction of clean documents unnecessarily flagged
  - Cost saved vs. judge-only (using DeepSeek as reference judge)

Uses the N-gram N=5 scanner as the gate (best AUC across domains).

Output: results/asr_threshold_analysis.json

Usage:
    uv run python scripts/eval_asr.py
"""

import json

import numpy as np

CHECKPOINT_PATH = "results/main_evaluation.json"
OPENROUTER_PATH = "results/openrouter_judges.json"
OUT_PATH = "results/asr_threshold_analysis.json"

# DeepSeek judge cost per domain (from openrouter_judges.json) — used for savings estimate
DEEPSEEK_COST = {
    "enron": 0.055485,
    "stackoverflow": 0.030743,
    "wikipedia": 0.166253,
}

ESCALATION_RATES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def threshold_for_rate(scores: list[float], rate: float) -> float:
    """Return the score threshold that flags exactly `rate` fraction of documents."""
    k = max(1, int(np.ceil(rate * len(scores))))
    return float(np.sort(scores)[-k])


def metrics_at_threshold(
    scores: list[float],
    labels: list[int],
    threshold: float,
) -> dict:
    flagged = [s >= threshold for s in scores]
    n_total = len(labels)
    n_poisoned = sum(labels)
    n_clean = n_total - n_poisoned

    tp = sum(f and lbl for f, lbl in zip(flagged, labels, strict=False))
    fp = sum(f and not lbl for f, lbl in zip(flagged, labels, strict=False))
    fn = sum(not f and lbl for f, lbl in zip(flagged, labels, strict=False))

    tpr = tp / n_poisoned if n_poisoned else 0.0
    fpr = fp / n_clean if n_clean else 0.0
    asr_proxy = fn / n_poisoned if n_poisoned else 0.0
    escalation_rate = sum(flagged) / n_total

    return {
        "threshold": round(threshold, 6),
        "escalation_rate": round(escalation_rate, 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "asr_proxy": round(asr_proxy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


if __name__ == "__main__":
    with open(CHECKPOINT_PATH) as f:
        checkpoint = json.load(f)

    results: dict = {"scanner_method": "ngram_n5", "domains": {}}

    for domain, d in checkpoint["domains"].items():
        labels = d["labels"]
        scores = d["raw_scores"]["ngram_n5"]
        judge_cost = DEEPSEEK_COST.get(domain, 0.0)
        n_docs = len(labels)

        domain_results = []
        for rate in ESCALATION_RATES:
            thresh = threshold_for_rate(scores, rate)
            m = metrics_at_threshold(scores, labels, thresh)
            m["cost_usd"] = round(rate * judge_cost, 6)
            m["cost_saved_usd"] = round((1 - rate) * judge_cost, 6)
            m["cost_saved_pct"] = round((1 - rate) * 100, 1)
            domain_results.append(m)

        results["domains"][domain] = domain_results

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table per domain
    for domain, rows in results["domains"].items():
        judge_cost = DEEPSEEK_COST.get(domain, 0.0)
        print(f"\n{domain.upper()}  (DeepSeek judge cost if 100% escalated: ${judge_cost:.4f})")
        print(
            f"  {'Escalate':>9} {'TPR':>7} {'ASR':>7} {'FPR':>7} {'Cost saved':>12} {'Cost $':>9}"
        )
        print("  " + "-" * 58)
        for m in rows:
            print(
                f"  {m['escalation_rate'] * 100:>8.0f}%"
                f" {m['tpr'] * 100:>6.1f}%"
                f" {m['asr_proxy'] * 100:>6.1f}%"
                f" {m['fpr'] * 100:>6.1f}%"
                f" {m['cost_saved_pct']:>10.1f}%"
                f" ${m['cost_usd']:>8.4f}"
            )

    print(f"\nSaved to {OUT_PATH}")
