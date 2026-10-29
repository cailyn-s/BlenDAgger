"""Two ways to produce the numbers a figure needs.

    from_results(...)   an analysis run: the score files it wrote
    from_export(...)    the exported report: <output_dir>/export/*.csv

Both return the same shape, so downstream plotting code can be driven from
either side without knowing which one it got.

Coverage note. `rollouts.csv` carries one whole-task `success` per rollout (see
ROLLOUT_COLUMNS in episode_export.py) and no per-subtask breakdown, so
`from_export` can supply overall success but NOT the per-stage fractions the
stacked subtask figure draws. Reproducing that figure from the report would
need per-subtask columns added to the export, whose schema is fixed because
downstream statistics code depends on it. Use `from_results` for the stacked
figure; both work for overall success.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from collections import OrderedDict

from metrics.stats import _load_subtask_fractions, _n_episodes


def _score_path(folder: str) -> str | None:
    """The score file in an eval folder, whichever robot wrote it."""
    for name in ("subtask_scores.json", "eval_results.json"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    return None


def from_results(folders: list[str], n_subtasks: int) -> dict:
    """Per-stage fractions and rollout counts, straight from an analysis run.

    folders: eval folders, in the order the bars should appear.
    Returns {"fractions": [[...]], "n_eps": [...], "sources": [...]}.
    """
    fractions, n_eps, sources = [], [], []
    for folder in folders:
        p = folder if folder.endswith(".json") else _score_path(folder)
        if p is None:
            raise FileNotFoundError(f"no subtask_scores.json / eval_results.json under {folder!r}")
        fractions.append(_load_subtask_fractions(p, n_subtasks))
        n_eps.append(_n_episodes(p))
        sources.append(p)
    return {"fractions": fractions, "n_eps": n_eps, "sources": sources}


def from_export(export_dir: str) -> dict:
    """Whole-task success per (method, round) from the exported report.

    Reads `<export_dir>/rollouts.csv`. Returns
    {"success": {(method, round): fraction}, "n": {(method, round): count},
     "rows": [...]} -- enough for the success figures, not for the per-stage
    stack (see the module docstring).
    """
    path = os.path.join(export_dir, "rollouts.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no rollouts.csv under {export_dir!r}")
    hits: dict = OrderedDict()
    total: dict = OrderedDict()
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
            key = (row.get("method"), row.get("round"))
            total[key] = total.get(key, 0) + 1
            if str(row.get("success", "")).strip() in ("1", "True", "true"):
                hits[key] = hits.get(key, 0) + 1
    success = {k: hits.get(k, 0) / total[k] for k in total}
    return {"success": success, "n": dict(total), "rows": rows}


def export_covers_subtasks(export_dir: str) -> bool:
    """True if rollouts.csv gained per-subtask columns and can drive the stack."""
    path = os.path.join(export_dir, "rollouts.csv")
    if not os.path.exists(path):
        return False
    with open(path, newline="") as fh:
        header = next(csv.reader(fh), [])
    return any(c.startswith("subtask") for c in header)


# --- loading for the replot path ---


def load_saved_distributions(output_dir: str):
    """(meta, data) from a previous analysis run's ood/ directory."""
    npz_path = os.path.join(output_dir, "ood", "distributions.npz")
    meta_path = os.path.join(output_dir, "ood", "distributions_meta.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Distributions file not found: {npz_path}\n"
            "Run without plot_only=true first to generate it.")
    with open(meta_path) as fp:
        meta = json.load(fp)
    data = np.load(npz_path, allow_pickle=False)
    # The meta is hand-editable and the npz is not, so catch a label mismatch early.
    missing = [a for a in meta.get("algo_labels", [])
               if f"{a}__knn__all" not in data.files]
    if missing:
        present = sorted({k.split("__")[0] for k in data.files})
        raise KeyError(
            f"{meta_path} lists algo(s) {missing} that are absent from {npz_path}.\n"
            f"  npz contains: {present}\n"
            "  The two were written at different times -- rerun the analysis stage "
            "so both are regenerated, or align the label in the meta.")
    return meta, data
