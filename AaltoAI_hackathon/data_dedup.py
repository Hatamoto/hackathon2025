from __future__ import annotations
"""
AI-assisted merger (v2.3, 9 Jun 2025)

• ID-normalisointi ennen klusterointia (fuzzy-match → sama innovation_id)
• 2-tason LLM-varmistus:
    1) core_summary      2) tarvittaessa descriptions
• Blocking + RapidFuzz quick filter → LLM-kutsut vain relevantteihin pareihin
• Batch-kyselyt (≤20 paria)         → 5–10 × vähemmän round-tripejä
• Tulosteet: merged_innovations.json  &  singles.json
"""

import argparse
import json
import re
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from rapidfuzz import fuzz
from langchain_openai import AzureChatOpenAI

from utils import write_json_and_track

# ──────────────────────────────  Oletusarvot  ────────────────────────────────

DEFAULT_INPUT = "filtered_innovations.json"
DEFAULT_MERGED_OUT = "merged_innovations.json"
DEFAULT_SINGLES_OUT = "singles.json"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_BATCH_SIZE = 20
DEFAULT_FUZZY_THR = 40
ID_SIM_THRESHOLD = 80           # jos innovation_id-parin fuzz ≥80 % → kandidaatti
DESC_MAXLEN = 800          # tekstiä promptiin / kuvaus

# ─────────────────────────────  LLM-alustus  ────────────────────────────────

load_dotenv()
CFG_FILE = "azure_config.json"


def _load_cfg(deployment: str, path: str = CFG_FILE) -> Dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)[deployment]


def init_llm(deployment: str) -> AzureChatOpenAI:
    cfg = _load_cfg(deployment)
    return AzureChatOpenAI(
        model=deployment,
        api_key=cfg["api_key"],
        deployment_name=cfg["deployment"],  # type: ignore[arg-type]
        azure_endpoint=cfg["api_base"],
        api_version=cfg["api_version"],
    )

# ──────────────────────────────  Apufunktiot  ────────────────────────────────


def _slug(txt: str) -> str:
    return re.sub(r"\W+", "", (txt or "")).lower()[:40]


def _clean(txt: str, maxlen: int = 800) -> str:
    return re.sub(r"\s+", " ", txt or "").strip()[:maxlen]

# ─────────────────────────────  Union–Find  ──────────────────────────────────


class UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[rj] = ri

# ──────────────────  1) ID-normalisointi-passi  ──────────────────────────────


def normalise_ids(data: List[Dict]) -> None:
    """Jos uusi ID ~ samanlainen kuin aiempi, korvataan kanonisella."""
    canon: List[str] = []
    for rec in data:
        raw = rec.get("innovation_id") or ""
        best, score = None, 0
        for cid in canon:
            sc = fuzz.token_set_ratio(raw, cid)
            if sc > score:
                best, score = cid, sc
        if best and score >= ID_SIM_THRESHOLD:
            rec["innovation_id"] = best
        else:
            canon.append(raw)

# ────────────────────── 2) LLM-pohjainen vertailu  ───────────────────────────


def _llm_batch(
    pairs: List[Tuple[int, int]],
    entries: List[Dict],
    llm: AzureChatOpenAI,
    mode: str = "summary",
) -> List[str]:
    """Palauttaa YES/NO/UNSURE listan. mode = summary | desc"""
    if mode == "summary":
        header = ("For every numbered pair, answer ONLY YES, NO or UNSURE "
                  "(e.g. '1 YES'). Decide *solely* from the two summaries.")

        def left(i): return _clean(entries[i]["core_summary"])
        def right(j): return _clean(entries[j]["core_summary"])
    else:  # descriptions
        header = ("For every numbered pair, answer ONLY YES, NO or UNSURE "
                  "(e.g. '1 YES'). Decide from the two text snippets.")

        def _first_desc(idx: int) -> str:
            d = entries[idx].get("descriptions", [])
            return _clean(d[0]["text"] if d else "", DESC_MAXLEN)

        def left(i): return _first_desc(i)
        def right(j): return _first_desc(j)

    lines = []
    for k, (i, j) in enumerate(pairs, 1):
        lines.append(f"{k}. {left(i)} ### {right(j)}")
    prompt = header + "\n\n" + "\n".join(lines)

    resp = llm.invoke(prompt).content.strip().splitlines()  # type: ignore
    out: List[str] = []
    for line in resp:
        token = line.strip().split()[-1].upper()
        out.append(token if token in {"YES", "NO", "UNSURE"} else "UNSURE")
    while len(out) < len(pairs):
        out.append("UNSURE")
    return out

# ───────────────────────────── 3) Klusterointi  ──────────────────────────────


def cluster(
    entries: List[Dict],
    llm: AzureChatOpenAI,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fuzzy_thr: int = DEFAULT_FUZZY_THR,
) -> Tuple[List[List[int]], List[int]]:
    n = len(entries)
    uf = UF(n)

    # 3.1 Blocking slug
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(entries):
        buckets[_slug(rec["innovation_id"])].append(idx)

    # 3.2 Nopean fuzzy-suotimen parit
    candidate_pairs: List[Tuple[int, int]] = []
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            if fuzz.token_set_ratio(
                entries[i]["core_summary"], entries[j]["core_summary"]
            ) >= fuzzy_thr:
                candidate_pairs.append((i, j))

    print(f"🧮  LLM-vertailtavia pareja: {len(candidate_pairs):,}")

    # 3.3 1. erä – summary
    for start in range(0, len(candidate_pairs), batch_size):
        batch = candidate_pairs[start:start + batch_size]
        labs = _llm_batch(batch, entries, llm, mode="summary")

        # kerää UNSURE-parit fallbackia varten
        unsure_pairs = [p for p, l in zip(batch, labs) if l == "UNSURE"]

        # yhdistä YES
        for (i, j), lab in zip(batch, labs):
            if lab == "YES":
                uf.union(i, j)

        if not unsure_pairs:
            continue  # nopea

        # 3.4 2. erä – descriptions
        desc_labels = _llm_batch(unsure_pairs, entries, llm, mode="desc")
        for (i, j), lab in zip(unsure_pairs, desc_labels):
            if lab == "YES":
                uf.union(i, j)

    # 3.5 Ryhmittely tulokseksi
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        groups[uf.find(idx)].append(idx)

    clusters = [g for g in groups.values() if len(g) > 1]
    singles = [g[0] for g in groups.values() if len(g) == 1]

    return clusters, singles


def merge_cluster(idxs: List[int], data: List[Dict]) -> Dict:
    base = data[idxs[0]].copy()
    for k in idxs[1:]:
        rec = data[k]
        seen = {tuple(p) for p in base.get("participants", [])}
        for p in rec.get("participants", []):
            if tuple(p) not in seen:
                base.setdefault("participants", []).append(p)
                seen.add(tuple(p))
        base.setdefault("descriptions", []).extend(rec.get("descriptions", []))
    return base

# ───────────────────────────  Pipeline driver  ───────────────────────────────


def run_deduplication_pipeline(
    input_path:  str = DEFAULT_INPUT,
    merged_out:  str = DEFAULT_MERGED_OUT,
    singles_out: str = DEFAULT_SINGLES_OUT,
    model_name:  str = DEFAULT_MODEL,
    batch_size:  int = DEFAULT_BATCH_SIZE,
) -> None:
    llm = init_llm(model_name)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    normalise_ids(data)                           # 1) ID-passi
    clusters, singles = cluster(data, llm, batch_size=batch_size)  # 2–3

    merged = [merge_cluster(c, data) for c in clusters]
    singles_ = [data[i] for i in singles]

    write_json_and_track(merged_out, merged)
    write_json_and_track(singles_out, singles_)

    print("\n✅  Dedup-pipeline valmis")
    print("   Entries käsitelty :", len(data))
    print("   Klustereita       :", len(clusters))
    print("   Yhdistettyjä      :", sum(len(c) for c in clusters))
    print("   Singles           :", len(singles_))
    print("   →", merged_out, "/", singles_out)


# Alias vanhoille importeille
deduplicate_innovations = run_deduplication_pipeline

# ───────────────────────────────  CLI  ───────────────────────────────────────


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Merge innovation entries via 2-phase LLM")
    p.add_argument("--input",      default=DEFAULT_INPUT)
    p.add_argument("--merged_out", default=DEFAULT_MERGED_OUT)
    p.add_argument("--singles_out", default=DEFAULT_SINGLES_OUT)
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                   help="Pair batch-size per LLM call")
    args = p.parse_args()

    run_deduplication_pipeline(
        input_path=args.input,
        merged_out=args.merged_out,
        singles_out=args.singles_out,
        model_name=args.model,
        batch_size=args.batch,
    )
