#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_cluster.py  (one-click v6.0 – 1 Jun 2025)

Uutta v5.1 → v6.0
──────────────────
✔  Klustereiden jäsenet yhdistetään *yhdeksi* entryksi
   ja tallennetaan **suoraan** `filtered_innovations.json`-tiedostoon.
✔  Sinkut kopioidaan sellaisenaan.
✔  `text_clusters.json` ja `text_singles.json` luodaan enää vain
   DEBUG-tarpeisiin – poistuvat ajon lopuksi.

Tiedostovirta
─────────────
filtered_innovations.json  → (ID:n lisäys) →  innovations_with_ids.json
                                 │
                                 ├─ Klusterointi & yhdistäminen
                                 ▼
                         filtered_innovations.json   (päivitetty, deduplikoitu)
"""

from __future__ import annotations
import os, sys, json, re, gzip, pathlib, unicodedata, traceback
from typing import Dict, List
import ndjson, numpy as np, faiss
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

# ─────────── asetukset & polut ───────────────────────────────────────────
SRC_JSON        = "filtered_innovations.json"
JSON_WITH_IDS   = "innovations_with_ids.json"
TEXTS_NDJSONGZ  = "entry_texts.ndjson.gz"
CLUSTER_JSON    = "text_clusters.json"   # debug/temporary
SINGLES_JSON    = "text_singles.json"    # debug/temporary

CHAT_MODEL      = "gpt-4o-mini"
EMBED_MODEL     = "text-embedding-3-large"
K_NEIGH         = 10
COS_THR         = 0.78
CFG_FILE        = "azure_config.json"

# ─────────── UTF-8 I/O ───────────────────────────────────────────────────
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # Py<3.7
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────── yleisiä apuja ───────────────────────────────────────────────
def normalize_and_strip(txt: str) -> str:
    return re.sub(r"[^\x00-\x7F]", "", unicodedata.normalize("NFKD", txt or ""))

def ascii_only(txt: str) -> str:
    return normalize_and_strip(txt).encode("ascii", "ignore").decode("ascii")

# ─────────── Azure-apu ───────────────────────────────────────────────────
def _load_cfg(deployment: str) -> Dict[str, str]:
    with open(CFG_FILE, encoding="utf-8") as fp:
        return {k: ascii_only(v) for k, v in json.load(fp)[deployment].items()}

def init_chat_llm(name: str = CHAT_MODEL) -> AzureChatOpenAI:
    c = _load_cfg(name)
    return AzureChatOpenAI(
        model=name, api_key=c["api_key"],
        deployment_name=c["deployment"], azure_endpoint=c["api_base"],
        api_version=c["api_version"]
    )

def init_embed_client(name: str = EMBED_MODEL) -> tuple[AzureOpenAI, str]:
    c = _load_cfg(name)
    cli = AzureOpenAI(
        api_key=c["api_key"], azure_endpoint=c["api_base"],
        api_version=c["api_version"]
    )
    return cli, c["deployment"]

# ─────────── 1) lisää entry_id:t ─────────────────────────────────────────
def add_ids() -> List[Dict]:
    data = json.loads(pathlib.Path(SRC_JSON).read_text(encoding="utf-8"))
    for i, rec in enumerate(data, 1):
        rec["entry_id"] = f"E{i:05d}"
    pathlib.Path(JSON_WITH_IDS).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅  {len(data)} entryä → {JSON_WITH_IDS}")
    return data

# ─────────── 2) tekstien vienti ──────────────────────────────────────────
def concat_text(rec: Dict, max_len: int = 1_000) -> str:
    return re.sub(r"\s+", " ", f"{rec.get('core_summary','')} {rec.get('descriptions',[{}])[0].get('text','')}".strip())[:max_len]

def export_texts(data: List[Dict]) -> None:
    with gzip.open(TEXTS_NDJSONGZ, "wt", encoding="utf-8") as fp:
        w = ndjson.writer(fp, ensure_ascii=False)
        for r in data:
            w.writerow({"entry_id": r["entry_id"], "text": ascii_only(concat_text(r))})
    print(f"✅  {len(data)} riviä → {TEXTS_NDJSONGZ}")

# ─────────── 3) klusteroi & yhdistä ──────────────────────────────────────
def merge_cluster(idxs: List[int], all_rec: List[Dict]) -> Dict:
    """Yhdistää idxs-listan entryt yhdeksi tietueeksi."""
    base = json.loads(json.dumps(all_rec[idxs[0]]))  # deep-copy
    for k in idxs[1:]:
        rec = all_rec[k]
        # yhdistele osallistujat
        seen = {tuple(p) for p in base.get("participants", [])}
        for p in rec.get("participants", []):
            if tuple(p) not in seen:
                base.setdefault("participants", []).append(p); seen.add(tuple(p))
        # yhdistele kuvaukset
        base.setdefault("descriptions", []).extend(rec.get("descriptions", []))
    return base

def deduplicate_and_save(data: List[Dict]) -> None:
    # --- (a) Lataa tekstit ↦ embeddit ↦ klusterit ----
    entry_ids, texts = [], []
    with gzip.open(TEXTS_NDJSONGZ, "rt", encoding="utf-8") as fp:
        for obj in ndjson.reader(fp):
            entry_ids.append(obj["entry_id"]); texts.append(obj["text"])
    n = len(texts); id2text = dict(zip(entry_ids, texts))

    emb_cli, emb_depl = init_embed_client()
    EMB_BATCH, embs, buf = 100, [], []
    def flush():
        if buf:
            resp = emb_cli.embeddings.create(model=emb_depl, input=buf)
            embs.extend([d.embedding for d in resp.data]); buf.clear()
    for t in tqdm(texts, desc="Embeddings", ascii=True):
        buf.append(t);  flush() if len(buf)>=EMB_BATCH else None
    flush()

    mat = np.asarray(embs, dtype="float32"); faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1]); index.add(mat)

    parent = list(range(n))
    def find(i):  # path compression
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb: parent[rb]=ra

    chat = init_chat_llm()

    for i in tqdm(range(n), desc="Neighbours", ascii=True):
        D,I = index.search(mat[i:i+1], K_NEIGH)
        for d,j in zip(D[0][1:], I[0][1:]):
            if d < COS_THR: break
            if find(i)==find(j): continue
            if d>=0.95: union(i,j); continue
            prompt = f"Answer YES or NO only. Are these two paragraphs about the SAME innovation?\n\nA: {texts[i]}\n\nB: {texts[j]}"
            if chat.invoke(prompt).content.strip().upper().startswith("YES"):
                union(i,j)

    groups: Dict[int,List[int]] = {}
    idx_by_id = {eid:i for i,eid in enumerate(entry_ids)}
    for idx,eid in enumerate(entry_ids):
        groups.setdefault(find(idx), []).append(idx)

    clusters_idx = [g for g in groups.values() if len(g)>1]
    singles_idx  = [g[0] for g in groups.values() if len(g)==1]

    # --- (b) Yhdistä tietueet listaksi ----------------
    merged: List[Dict] = []
    for g in clusters_idx:
        merged.append( merge_cluster(g, data) )
    for idx in singles_idx:
        merged.append( data[idx] )

    # --- (c) Kirjoita suoraan filtered_innovations.json ----
    pathlib.Path(SRC_JSON).write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾  Päivitetty {SRC_JSON}:  klustereita {len(clusters_idx)} | singles {len(singles_idx)} | yhteensä {len(merged)}")

    # --- (d) DEBUG-tiedostot -------------
    pathlib.Path(CLUSTER_JSON).write_text(
        json.dumps([[{"entry_id": entry_ids[i],
                      "text": id2text[entry_ids[i]]} for i in g]
                    for g in clusters_idx],
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    pathlib.Path(SINGLES_JSON).write_text(
        json.dumps([{"entry_id": entry_ids[i],
                     "text": id2text[entry_ids[i]]} for i in singles_idx],
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    # Poistetaan vain tilapäiset tiedostot,
    # mutta jätetään singles.json talteen
    for tmp in (CLUSTER_JSON, JSON_WITH_IDS, TEXTS_NDJSONGZ):
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass

# ─────────── one-click pipeline ──────────────────────────────────────────
if __name__ == "__main__":
    try:
        load_dotenv()
        all_data = add_ids()           # 1) ID:t
        export_texts(all_data)         # 2) Tekstit NDJSON.gz
        deduplicate_and_save(all_data) # 3) Klusterit → yhdistä → päivitä SRC_JSON
        print("✅  Valmis!")

    except Exception:
        print("❌  Virhe:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)