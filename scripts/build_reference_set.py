"""Build a principled reference threat set from JailbreakBench.

Selects one canonical behavior per JBB category (10 total), giving the
scanner a citable, taxonomy-grounded reference set rather than hand-written
phrases. The selected behaviors are the shortest, most direct statement of
intent in each category — best anchor points in embedding space.

Output: data/reference_set.json
"""

import json

from datasets import load_dataset

JBB_DATASET = "JailbreakBench/JBB-Behaviors"
OUT_PATH = "data/reference_set.json"

# One canonical behavior per category — chosen to be direct and unambiguous
# so the embedding anchor is as clean as possible. Adjust per domain as needed.
CANONICAL_INDEX_PER_CATEGORY: dict[str, int | None] = {
    "Disinformation": None,  # pick shortest in category
    "Economic harm": None,
    "Expert advice": None,
    "Fraud/Deception": None,
    "Government decision-making": None,
    "Harassment/Discrimination": None,
    "Malware/Hacking": None,
    "Physical harm": None,
    "Privacy": None,
    "Sexual/Adult content": None,
}


def select_reference(behaviors: list[dict]) -> dict:
    """From a list of behaviors in one category, pick the shortest Goal.

    Shortest Goal = most atomic statement of intent = best embedding anchor.
    """
    return min(behaviors, key=lambda b: len(b["Goal"]))


if __name__ == "__main__":
    print(f"Loading {JBB_DATASET}...")
    ds = load_dataset(JBB_DATASET, "behaviors", split="harmful")
    print(f"Loaded {len(ds)} harmful behaviors across categories.")

    # Group by category
    by_category: dict[str, list[dict]] = {}
    for row in ds:
        by_category.setdefault(row["Category"], []).append(row)

    print(f"\nCategories found ({len(by_category)}):")
    for cat, rows in sorted(by_category.items()):
        print(f"  {cat}: {len(rows)} behaviors")

    # Select one per category
    reference_set = []
    for category in sorted(by_category.keys()):
        chosen = select_reference(by_category[category])
        reference_set.append(
            {
                "category": category,
                "goal": chosen["Goal"],
                "source": chosen["Source"],
                "jbb_index": chosen["Index"],
            }
        )

    print(f"\nSelected {len(reference_set)} reference threats:")
    for r in reference_set:
        print(f"  [{r['category']}] {r['goal']}")

    output = {
        "source_dataset": JBB_DATASET,
        "selection_strategy": "shortest Goal per category (most atomic statement of intent)",
        "n_categories": len(reference_set),
        "reference_set": reference_set,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
