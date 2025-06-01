from __future__ import annotations

"""AI‑assisted merger of innovation entries (chunked version)

Processes the input JSON in **chunks of 10 records** so that we never feed a
huge comparison matrix to the LLM at once. After every chunk the script prints
progress in real time.

Outputs
=======
• ``merged_innovations.json``  – aggregated records (≥2 sources)
• ``singles.json``             – entries that stayed solitary

Run example
-----------
::

    python innovation_merger.py \
        --input filtered_innovations.json \
        --chunk 10 \
        --model gpt-4.1-mini
"""

import argparse
import json
import os
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# ────────────────────────────────────────────────────────────────────────────────
# Helpers: configuration & LLM init
# ────────────────────────────────────────────────────────────────────────────────

load_dotenv()

CONFIG_FILE = "azure_config.json"


def load_config(path: str = CONFIG_FILE) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def init_llm(deployment: str, config_path: str = CONFIG_FILE) -> AzureChatOpenAI:
    cfg = load_config(config_path)[deployment]
    return AzureChatOpenAI(
        model=deployment,
        api_key=cfg["api_key"],
        deployment_name=cfg["deployment"],  # type: ignore[arg-type]
        azure_endpoint=cfg["api_base"],
        api_version=cfg["api_version"],
    )

# ────────────────────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────────────────────


def clean(text: str) -> str:
    """Normalise whitespace; shrink very long text for prompts."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:1500]  # keep prompt under control

# ────────────────────────────────────────────────────────────────────────────────
# LLM based comparators
# ────────────────────────────────────────────────────────────────────────────────


def _llm_binary(question: str, model: AzureChatOpenAI) -> str:
    """Ask LLM a STRICT YES / NO / UNSURE question."""
    prompt = "Answer STRICTLY with YES, NO, or UNSURE (no extra words).\n" + question
    reply = model.invoke(prompt).content.strip().upper()
    if reply.startswith("YES"):
        return "YES"
    if reply.startswith("NO"):
        return "NO"
    return "UNSURE"


def same_by_core_summary(a: Dict, b: Dict, model: AzureChatOpenAI) -> str:
    q = (
        "Are these two core summaries about the identical innovation?\n\n"
        f"1. {clean(a['core_summary'])}\n\n2. {clean(b['core_summary'])}"
    )
    return _llm_binary(q, model)


def same_by_descriptions(a: Dict, b: Dict, model: AzureChatOpenAI) -> str:
    desc_a = " ".join(clean(d["text"]) for d in a.get("descriptions", [])[:2])
    desc_b = " ".join(clean(d["text"]) for d in b.get("descriptions", [])[:2])
    q = (
        "Are these two paragraphs describing the same innovation?\n\n"
        f"PARAGRAPH A:\n{desc_a}\n\nPARAGRAPH B:\n{desc_b}"
    )
    return _llm_binary(q, model)


def is_same_innovation(a: Dict, b: Dict, model: AzureChatOpenAI) -> bool:
    """Two‑stage decision: core summary first; if UNSURE then use descriptions."""
    first = same_by_core_summary(a, b, model)
    if first == "YES":
        return True
    if first == "NO":
        return False
    # fallback: use richer context
    second = same_by_descriptions(a, b, model)
    return second == "YES"

# ────────────────────────────────────────────────────────────────────────────────
# Core clustering logic (chunk‑aware)
# ────────────────────────────────────────────────────────────────────────────────


def cluster(
    entries: List[Dict],
    model: AzureChatOpenAI,
    chunk_size: int = 10,
) -> Tuple[List[List[int]], List[int]]:
    """Union–Find clustering with progress printed per *chunk_size* items."""
    n = len(entries)
    parent = list(range(n))  # Union‑Find disjoint‑set

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    total_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_idx, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        print(
            f"Processing chunk {chunk_idx + 1}/{total_chunks} (records {start}-{end - 1})…",
            flush=True,
        )

        # Compare every record in current chunk against ALL *following* records
        for i in range(start, end):
            for j in range(i + 1, n):  # includes future chunks → full coverage
                if find(i) == find(j):
                    continue  # already same cluster
                if is_same_innovation(entries[i], entries[j], model):
                    union(i, j)

    # Build cluster list / singles list
    clusters_map: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        clusters_map[find(idx)].append(idx)

    clusters = [members for members in clusters_map.values() if len(members) > 1]
    singles = [members[0] for members in clusters_map.values() if len(members) == 1]
    return clusters, singles


def merge_cluster(cluster_indices: List[int], entries: List[Dict]) -> Dict:
    """Aggregate multiple source entries into one."""
    base = entries[cluster_indices[0]].copy()
    for idx in cluster_indices[1:]:
        rec = entries[idx]
        # union participants
        seen = {tuple(p) for p in base.get("participants", [])}
        for p in rec.get("participants", []):
            if tuple(p) not in seen:
                base.setdefault("participants", []).append(p)
                seen.add(tuple(p))
        # extend descriptions
        base.setdefault("descriptions", []).extend(rec.get("descriptions", []))
    return base

# ────────────────────────────────────────────────────────────────────────────────
# Pipeline driver
# ────────────────────────────────────────────────────────────────────────────────


def run_deduplication_pipeline(
    input_path: str,
    merged_out: str,
    singles_out: str,
    model_name: str,
    chunk_size: int,
) -> None:
    llm = init_llm(model_name)

    with open(input_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    clusters, singles_idx = cluster(data, llm, chunk_size)

    merged_records = [merge_cluster(c, data) for c in clusters]
    single_records = [data[i] for i in singles_idx]

    with open(merged_out, "w", encoding="utf-8") as fp:
        json.dump(merged_records, fp, indent=2, ensure_ascii=False)
    with open(singles_out, "w", encoding="utf-8") as fp:
        json.dump(single_records, fp, indent=2, ensure_ascii=False)

    # ── stats ────────────────────────────────────────────────────────────
    print("Processed entries:", len(data))
    print("Clusters formed  :", len(clusters))
    print("Records merged   :", sum(len(c) for c in clusters))
    print("Singles left     :", len(single_records))
    print("→", merged_out, "/", singles_out)

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge innovation entries via LLM (chunked)")
    parser.add_argument("--input", default="filtered_innovations.json")
    parser.add_argument("--merged_out", default="merged_innovations.json")
    parser.add_argument("--singles_out", default="singles.json")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--chunk", type=int, default=10, help="Number of records per chunk")
    args = parser.parse_args()

    run(args.input, args.merged_out, args.singles_out, args.model, args.chunk)
