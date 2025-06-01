# data_loop.py – Iterative deduplication helper
# --------------------------------------------------
# Runs the existing deduplication + text‑cluster + LLM‑validation cycle until
# the LLM step no longer rejects any entries (or until a safety iteration cap
# is reached).  In addition, we now *append* to filtered_innovations_rejected.json
# instead of overwriting it at every pass – so you keep a cumulative log of
# everything the model has thrown out during the whole run.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

REJECTED_PATH = Path("filtered_innovations_rejected.json")

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _load_json_list(path: Path) -> list[Any]:
    """Load a JSON file that *must* contain a top‑level list. Return [] if missing."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json_list(path: Path, data: list[Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _merge_rejected(old: list[Any], new: list[Any]) -> list[Any]:
    """Return *old* ∪ *new* with deduplication (order preserved as in *old*→*new*)."""
    seen: set[str] = {json.dumps(item, sort_keys=True) for item in old}
    merged = list(old)
    for item in new:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _count_json(path: Path) -> int:
    """Return list length for *path* (0 if missing)."""
    return len(_load_json_list(path))


def _print_stats(iter_n: int) -> tuple[int, int]:
    """Pretty‑print and return keep/reject counts after each iteration."""
    kept = _count_json(Path("filtered_innovations.json"))
    rejected = _count_json(REJECTED_PATH)
    print(f"📊 Iteraatio {iter_n}: jäljellä {kept} • poistettu yhteensä {rejected}")
    return kept, rejected

# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def dedup_until_converged(max_iter: int = 10, *, run_cluster: bool = True) -> None:
    """Run the full dedup‑validation loop until the LLM no longer rejects rows.

    Parameters
    ----------
    max_iter : int, optional
        Hard safety stop so we never loop forever (default 10).
    run_cluster : bool, optional
        Whether to invoke data_cluster.py between passes (default True).
    """

    # Delayed import to avoid circular dependency if you inline this function
    from main import (
        run_deduplication_pipeline,
        run_final_validation,
    )

    prev_kept: Optional[int] = None

    for i in range(1, max_iter + 1):
        print(f"\n⚙️  Dedup‑loop kierros {i}/{max_iter}")

        # Snapshot current cumulative rejects BEFORE new run
        cumulative_rejects_before = _load_json_list(REJECTED_PATH)

        # 1) Rule‑based + text‑cluster deduplication
        run_deduplication_pipeline()

        # 2) Optional extra semantic clustering pass between rounds
        if run_cluster:
            print("\n🚀  Suoritetaan tekstiklusterointi (data_cluster.py)…")
            subprocess.run([sys.executable, "data_cluster.py"], check=True)
            print("✅  Tekstiklusterointi valmis")

        # 3) LLM validation step (this overwrites filtered_innovations_rejected.json)
        run_final_validation()

        # 4) Append new rejects to cumulative file
        latest_rejects = _load_json_list(REJECTED_PATH)
        merged_rejects = _merge_rejected(cumulative_rejects_before, latest_rejects)
        if len(merged_rejects) != len(latest_rejects):
            _save_json_list(REJECTED_PATH, merged_rejects)

        # 5) Check convergence criteria
        kept, rejected = _print_stats(i)
        if rejected == len(cumulative_rejects_before):  # no *new* rejects added
            print("🎉  Konvergenssi saavutettu – LLM ei poistanut enää uusia rivejä.")
            break

        # Secondary stop: list of kept rows unchanged ⇒ stabilised
        if prev_kept is not None and kept == prev_kept:
            print("ℹ️  Jäljelle jääneiden määrä pysyi samana. Pysäytetään silmukka varmuuden vuoksi.")
            break

        prev_kept = kept

    else:
        print(
            f"⚠️  Saavuttiin max_iter={max_iter}. Tarkista tulokset – rejected‑määrä kasvaa yhä."
        )

# ---------------------------------------------------------------------------
#  Script entry‑point convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dedup_until_converged()