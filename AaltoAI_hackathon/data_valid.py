#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_validate_innovations.py  –  viimeinen validointivaihe (v1.1, 1 Jun 2025)
───────────────────────────────────────────────────────────────────────────────
• Lukee klusteroidun & puhdistetun **filtered_innovations.json**‑tiedoston.
• GPT‑luokitus + Wikipedian innovaatiomääritelmä:
      YES  → todellinen innovaatio  → mukana *filtered_innovations_final.json*‑tiedostossa
      NO   → tapahtuma/uutinen     → talletetaan erilliseen *filtered_innovations_rejected.json*‑tiedostoon
• Kaikki muut tiedostot (esim. singles.json) jätetään ennalleen.

CLI‑vaihtoehdot
────────────────
--model   Azure‑deployment (oletus gpt-4o-mini)
--input   lähde‑JSON (oletus filtered_innovations.json)
--output  säilytetyt (oletus filtered_innovations_final.json)
--reject  poistetut  (oletus filtered_innovations_rejected.json)

Vaatii samat kirjastot ja avainasetukset kuin muutkin skriptit.
"""
from __future__ import annotations

import argparse, json, os, sys, traceback, requests
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# ─────────── asetukset ────────────────────────────────────────────────
DEFAULT_MODEL   = "gpt-4o-mini"
DEFAULT_INPUT   = "filtered_innovations.json"
#DEFAULT_OUTPUT  = "filtered_innovations_final.json"
DEFAULT_OUTPUT  = "filtered_innovations.json"
DEFAULT_REJECT  = "filtered_innovations_rejected.json"
CFG_FILE        = "azure_config.json"

# ─────────── Wikipedian määritelmä ────────────────────────────────────
try:
    _WIKI_DEFINITION = requests.get(
        "https://fi.wikipedia.org/api/rest_v1/page/summary/Innovaatio", timeout=10
    ).json().get("extract", "")
except Exception:
    _WIKI_DEFINITION = (
        "Innovaatio on uusi tai olennaisesti parannettu tuote, palvelu, prosessi "
        "tai menetelmä, joka otetaan käyttöön ja tuottaa lisäarvoa."
    )

# ─────────── Azure‑apu --------------------------------------------------

def _load_cfg(deployment: str, path: str = CFG_FILE) -> Dict[str, str]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))[deployment]
    env_key  = os.getenv(f"AZURE_OPENAI_API_KEY_{deployment.upper().replace('.','').replace('-','_')}")
    env_base = os.getenv(f"AZURE_OPENAI_BASE_URL_{deployment.upper().replace('.','').replace('-','_')}")
    if env_key:  cfg["api_key"]  = env_key
    if env_base: cfg["api_base"] = env_base
    return cfg


def init_llm(dep: str) -> AzureChatOpenAI:
    c = _load_cfg(dep)
    return AzureChatOpenAI(
        model=dep, api_key=c["api_key"], deployment_name=c["deployment"],
        azure_endpoint=c["api_base"], api_version=c["api_version"]
    )

# ─────────── LLM‑luokitusfunktio --------------------------------------

def is_true_innovation(summary: str, texts: List[str], llm: AzureChatOpenAI) -> bool:
    """True = teksti kuvaa varsinaista innovaatiota, False = uutinen/tapahtuma tms."""

    prompt = f"""Vastaa AINOASTAAN YES tai NO.

Wikipedian innovaatiomääritelmä:
\"{_WIKI_DEFINITION}\"

Arvioi, kuvaavatko seuraava yhteenveto ja tekstikatkelmat todellista teknologista tai tieteellistä INNOVAATIOTA (tuote, prototyyppi, prosessi, algoritmi) vai lähinnä tapahtumaa, rahoitusuutista tms.

YHTEENVETO:
\"\"\"
{summary.strip()}
\"\"\"

TEKSTIT (max 3):
\"\"\"
{chr(10).join(texts[:3]).strip()}
\"\"\"

Vastaa:"""

    try:
        ans = llm.invoke(prompt).content.strip().upper()
        return ans.startswith("Y") or ans.startswith("YES")
    except Exception as e:
        print(f"⚠️  LLM‑kutsu epäonnistui: {e}", file=sys.stderr)
        return True  # fallback: säilytä

# ─────────── päätoiminto ---------------------------------------------- ----------------------------------------------

def main(model: str, infile: str, outfile: str, rejectfile: str) -> None:
    load_dotenv()
    recs: List[Dict[str, Any]] = json.loads(Path(infile).read_text(encoding="utf-8"))
    llm = init_llm(model)

    kept:   List[Dict[str, Any]] = []
    reject: List[Dict[str, Any]] = []

    for rec in tqdm(recs, desc="LLM innovation check"):
        texts = [d.get("text", "") for d in rec.get("descriptions", [])]
        if is_true_innovation(rec.get("core_summary", ""), texts, llm):
            kept.append(rec)
        else:
            reject.append(rec)

    Path(outfile).write_text(json.dumps(kept,   indent=2, ensure_ascii=False), encoding="utf-8")
    Path(rejectfile).write_text(json.dumps(reject, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"🎯 LLM‑tarkistus: {len(kept):,}/{len(recs):,} säilytetty → {outfile} | "
        f"{len(reject):,} poistettu → {rejectfile}"
    )

# ─────────── Helper for programmatic use ------------------------------

def run_final_validation(
    model: str = DEFAULT_MODEL,
    infile: str = DEFAULT_INPUT,
    outfile: str = DEFAULT_OUTPUT,
    rejectfile: str = DEFAULT_REJECT,
) -> None:
    """Kutsu tämä suoraan esim. main.py:stä ilman CLI‑argumentteja."""
    main(model, infile, outfile, rejectfile)

# ─────────── CLI -------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM‑pohjainen innovaatiotarkistus")
    ap.add_argument("--model",  default=DEFAULT_MODEL,  help="Azure GPT‑deployment")
    ap.add_argument("--input",  default=DEFAULT_INPUT,  help="Sisääntulotiedosto")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Säilytettyjen JSON")
    ap.add_argument("--reject", default=DEFAULT_REJECT, help="Poistettujen JSON")
    args = ap.parse_args()

    try:
        main(args.model, args.input, args.output, args.reject)
    except Exception:
        print("❌  Virhe suorituksessa", file=sys.stderr)
        traceback.print_exc(); sys.exit(1)