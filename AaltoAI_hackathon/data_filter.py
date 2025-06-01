#!/usr/bin/env python
# coding: utf-8
"""
Updated data‑cleaning script (v1.6, 1 Jun 2025)
–––––––––––––––––––––––––––––––––––––––––––––––
• Cleans HTML/Markdown, trims extreme lengths, normalises dates.
• Canonical alias normalisation via VAT glossary + deduplicates participants.
• Global duplicate removal of identical description texts.
• Innovation article check (keywords → optional Azure OpenAI YES/NO).  
  – USE_LLM_CLASSIFIER=true enables LLM fallback.
• Prints detailed stats: total innovations processed, kept, removed, description counts, alias changes.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from bs4 import BeautifulSoup, FeatureNotFound
from dateutil import parser as dtparser
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# OPTIONAL DEPENDENCIES ------------------------------------------------------
# ---------------------------------------------------------------------------
try:
    from ftfy import fix_text  # Unicode fix
except ModuleNotFoundError:  # fallback
    def fix_text(text: str) -> str:  # type: ignore
        return text
    print("⚠️  ftfy not found – Unicode fixes disabled", file=sys.stderr)

load_dotenv()

# ---------------------------------------------------------------------------
# CONSTANTS & REGEXES --------------------------------------------------------
# ---------------------------------------------------------------------------
COMMON_BOILER_REGEXES = [
    re.compile(r"Open mobile menu\s+Close mobile menu", re.I),
    re.compile(r"Accept (all )?cookies", re.I),
    re.compile(r"Back to top", re.I),
]
MIN_LEN = 80
MAX_LEN = 10_000

KEYWORDS = re.compile(
    r"\b(innovation|prototype|patent|quantum computer|new technology|breakthrough|demonstrator|pilot project|kvantti|innovaatio|prototyyppi)\b",
    re.I,
)

USE_LLM = os.getenv("USE_LLM_CLASSIFIER", "false").lower() == "true"
if USE_LLM:
    from langchain_openai import AzureChatOpenAI  # type: ignore
    _LLM: Optional[AzureChatOpenAI] = None

    def _get_llm() -> AzureChatOpenAI:
        global _LLM
        if _LLM is None:
            cfg = json.loads(Path("data/keys/azure_config.json").read_text())
            m = "gpt-4.1-mini"
            cfg[m]["api_key"] = os.getenv("AZURE_OPENAI_API_KEY_41_M")
            cfg[m]["api_base"] = os.getenv("AZURE_OPENAI_BASE_URL_41_M")
            _LLM = AzureChatOpenAI(
                model=m,
                api_key=cfg[m]["api_key"],
                deployment_name=cfg[m]["deployment"],  # type: ignore[arg-type]
                azure_endpoint=cfg[m]["api_base"],
                api_version=cfg[m]["api_version"],
            )
        return _LLM

# ---------------------------------------------------------------------------
# GLOSSARY FOR ALIASES -------------------------------------------------------
# ---------------------------------------------------------------------------
_GLOSSARY_PATH = Path("resolved_entity_glossary.json") if Path(
    "resolved_entity_glossary.json").is_file() else Path("data/entity_glossary/entity_glossary.json")
try:
    _glossary_raw: Dict[str, Dict[str, Any]] = json.loads(
        _GLOSSARY_PATH.read_text(encoding="utf-8"))
    CANONICAL_ALIAS = {vat: info.get("alias", [vat])[
        0] for vat, info in _glossary_raw.items()}
    ALIAS_LOOKUP = {alias.lower().strip(): vat for vat, info in _glossary_raw.items()
                    for alias in info.get("alias", [])}
except Exception as e:
    print(f"⚠️  Glossary load failed: {e}", file=sys.stderr)
    CANONICAL_ALIAS, ALIAS_LOOKUP = {}, {}

# ---------------------------------------------------------------------------
# UTILITIES -----------------------------------------------------------------
# ---------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    try:
        soup = BeautifulSoup(text, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(text, "html.parser")
    return fix_text(soup.get_text(separator=" "))


def _clean_text(raw: str) -> Optional[str]:
    t = _strip_html(raw)
    for rx in COMMON_BOILER_REGEXES:
        t = rx.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if MIN_LEN <= len(t) <= MAX_LEN else None


def normalise_date(date_str: str | None) -> str:
    if not date_str:
        return "unknown"
    try:
        return dtparser.parse(date_str).date().isoformat()
    except Exception:
        return "unknown"

# ---------------------------------------------------------------------------
# INNOVATION CLASSIFIER ------------------------------------------------------
# ---------------------------------------------------------------------------


def _is_innovation(summary: str, fallback_text: str | None = None) -> bool:
    if KEYWORDS.search(summary):
        return True
    if fallback_text and KEYWORDS.search(fallback_text):
        return True
    if not USE_LLM:
        return False

    parts = [
        "You are a classifier. Answer YES or NO.",
        "Does the following Finnish/English text primarily describe a technological or scientific INNOVATION (product, prototype, device, process, software, algorithm, etc.)?",
        "",
        "TEXT:",
        "\"\"\"",
        summary.strip() or "<empty>",
        "\"\"\"",
    ]
    if fallback_text:
        parts += ["", "ADDITIONAL CONTEXT:",
                  "\"\"\"", fallback_text[:800], "\"\"\""]
    parts.append("Respond ONLY with YES or NO.")
    prompt = "\n".join(parts)

    try:
        ans = _get_llm().invoke(prompt).content.strip().lower()  # type: ignore
        return ans.startswith("y")
    except Exception as e:
        print(f"⚠️  LLM innovation check failed: {e}", file=sys.stderr)
        return False

# ---------------------------------------------------------------------------
# PARTICIPANT NORMALISATION --------------------------------------------------
# ---------------------------------------------------------------------------


def _normalise_participants(parts: List[List[str]]) -> Tuple[List[List[str]], int]:
    changes = 0
    seen: Dict[str, List[str]] = {}
    for vat, name in parts:
        if vat == "(not found)":
            guess = ALIAS_LOOKUP.get(name.lower().strip())
            if guess:
                vat = guess
                changes += 1
        canon = CANONICAL_ALIAS.get(vat, name)
        if canon != name:
            changes += 1
        key = canon.lower()
        if key in seen:
            if seen[key][0] == "(not found)" and vat != "(not found)":
                seen[key] = [vat, canon]
                changes += 1
            else:
                changes += 1
            continue
        seen[key] = [vat, canon]
    return list(seen.values()), changes

# ---------------------------------------------------------------------------
# MAIN PROCESSOR -------------------------------------------------------------
# ---------------------------------------------------------------------------


def process_innovations(recs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int, int, int]:
    out, seen_desc = [], set()
    desc_before = desc_after = alias_changes = removed_articles = 0

    for rec in recs:
        first_text = rec.get("descriptions", [{}])[0].get(
            "text", "") if rec.get("descriptions") else ""
        if not _is_innovation(rec.get("core_summary", ""), first_text):
            removed_articles += 1
            continue

        new_descs: List[Dict[str, str]] = []
        for d in rec.get("descriptions", []):
            desc_before += 1
            cleaned = _clean_text(d.get("text", ""))
            if not cleaned or cleaned in seen_desc:
                continue
            if not _is_innovation("", cleaned):  # verify each description
                continue
            new_descs.append({"source": d.get(
                "source"), "text": cleaned, "date": normalise_date(d.get("date"))})
            seen_desc.add(cleaned)
            desc_after += 1

        if not new_descs:
            removed_articles += 1
            continue

        norm_parts, delta = _normalise_participants(
            rec.get("participants", []))
        alias_changes += delta
        out.append({
            "innovation_id": rec.get("innovation_id"),
            "core_summary": rec.get("core_summary"),
            "participants": norm_parts,
            "descriptions": new_descs,
        })

    return out, desc_before, desc_after, alias_changes, removed_articles

# ---------------------------------------------------------------------------
# FILE WRAPPER --------------------------------------------------------------
# ---------------------------------------------------------------------------


def filter_innovations_file(
    input_file: str = "structured_innovations.json", *, output_file: str = "filtered_innovations.json"
) -> None:
    """Load innovations → clean → write filtered file + print detailed stats."""
    recs = json.loads(Path(input_file).read_text(encoding="utf-8"))
    total_innov = len(recs)

    cleaned, before, after, alias_chg, removed = process_innovations(recs)
    kept = len(cleaned)

    # write output
    Path(output_file).write_text(json.dumps(
        cleaned, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"✅ {total_innov:,} innovations processed • {kept:,} kept • {removed:,} removed as non‑innovation "
        f"• descriptions {after:,}/{before:,} (removed {before-after:,}) "
        f"• participant names normalised {alias_chg:,} times → {output_file}"
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    filter_innovations_file()
