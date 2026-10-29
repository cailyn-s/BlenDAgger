#!/usr/bin/env python3
"""
Responsiveness column of the ablation table: how quickly the arbitrated command comes to match
what the operator is commanding, per method.

One value per intervention, pooled over rounds 1-3 of collection. The clock starts at the first
step the operator commands motion and stops the first step the arbitrated action is within 20% of
the commanded direction and 20% of the commanded speed (commanded space, translation and rotation).
Interventions where that never happens are counted in 'never reached' and left out of the median.

Usage
-----
    python analysis/ablation_responsiveness.py --config analysis/ablation_responsiveness
    python analysis/ablation_responsiveness.py --data-root /path/to/data --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import episode_export as export
from metrics import intervention_response as iresp
from metrics import task_time
from metrics.config import load_task_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 20% of the commanded direction (cos >= 0.8) and of the commanded speed.
COS_THRESH = 0.80
MATCH_TOL = 0.20
ROUNDS = (1, 2, 3)


def round_files(data_root: str, folder: str, rounds) -> list[str]:
    """The collection file of each requested round in one run folder."""
    out = []
    for n in rounds:
        cand = sorted(p for p in glob.glob(os.path.join(data_root, folder,
                                                        f"*round{n}_*_dataset_full.hdf5"))
                      if "_im128" not in p and "_trimmed" not in p)
        if cand:
            out.append(cand[0])
    return out


def load_meta(path: str) -> dict:
    """The round's meta.json, keyed by episode, flattened past its per-demo "summary"."""
    meta_path = path[: -len("_dataset_full.hdf5")] + "_meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as fh:
        raw = json.load(fh)
    return {k: (v.get("summary", v) if isinstance(v, dict) else v) for k, v in raw.items()}


def load_demos(path: str) -> list[dict]:
    """Collection episodes with the fields compute_response reads; aborted ones dropped."""
    meta = load_meta(path)
    demos = []
    with h5py.File(path, "r") as f:
        # hdf5 demos keep the source dataset's indices while meta counts from zero.
        names = export.canonical_names(list(f["data"]), list(meta)) if meta else {}
        for name in sorted(f["data"], key=lambda k: int(k.split("_")[-1])):
            grp = f["data"][name]
            if bool(grp.attrs.get("aborted", False)):
                continue
            if bool((meta.get(names.get(name, name)) or {}).get("aborted", False)):
                continue
            if "intervention_mask" not in grp or "eef_pos" not in grp:
                continue
            iv = grp["intervention_mask"][:].astype(bool)
            demos.append(dict(
                intervention_mask=iv,
                actions=grp["actions"][:].astype(np.float64),
                human_actions=grp["human_actions"][:].astype(np.float64),
                arm_pos=grp["eef_pos"][:].astype(np.float64),
                arm_quat=grp["eef_quat"][:].astype(np.float64),
                grip_window=task_time.grip_window(grp, len(iv)),
            ))
    return demos


def responsiveness(demos: list[dict], hz: float, gap: int, action_scale: float) -> dict:
    res = iresp.compute_response(demos, hz=hz, gap_threshold=gap, action_scale=action_scale,
                                 channel="both", min_commanding=1)
    reached = res["latency_cmd_match"]
    n_never = len(res["latency_cmd_match_censored"])
    n = len(reached) + n_never
    return dict(
        interventions=n,
        median_s=float(np.median(reached) / hz) if reached.size else np.nan,
        never_reached=n_never / n if n else np.nan,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", default="analysis/ablation_responsiveness",
                    help="conf/<name>.yaml under the repo root")
    ap.add_argument("--data-root", default=None,
                    help="folder holding the run folders (overrides the config)")
    ap.add_argument("--csv", default=None, help="also write the table here")
    args = ap.parse_args()

    cfg = load_task_config(os.path.join(REPO, "conf", f"{args.config}.yaml"))
    data_root = args.data_root or str(cfg.data_root)
    if not os.path.isabs(data_root):
        data_root = os.path.join(REPO, data_root)
    hz = float(cfg.control_hz)
    gap = max(1, int(round(float(cfg.intervention_timing.gap_seconds) * hz)))
    action_scale = float(cfg.collection_action_scale)

    task_time.set_control_hz(hz)
    iresp.MAX_LINEAR_VEL = float(cfg.max_linear_vel)
    iresp.MAX_ANGULAR_VEL = float(cfg.max_angular_vel)
    iresp.ANGULAR_FRAME = str(cfg.angular_frame)
    iresp.ANGULAR_FRAME_YAW = float(cfg.angular_frame_yaw)
    iresp.COS_THRESH, iresp.MATCH_TOL = COS_THRESH, MATCH_TOL

    rows = []
    for label, folder in cfg.runs.items():
        paths = round_files(data_root, folder, ROUNDS)
        if not paths:
            print(f"  [skip] {label}: no rounds {list(ROUNDS)} under {os.path.join(data_root, folder)}")
            continue
        demos = [d for p in paths for d in load_demos(p)]
        rows.append(dict(method=label, rounds=len(paths), episodes=len(demos),
                         **responsiveness(demos, hz, gap, action_scale)))

    print(f"\n{'Method':<22}{'episodes':>9}{'interventions':>15}{'median (s)':>12}{'never reached':>15}")
    for r in rows:
        print(f"{r['method']:<22}{r['episodes']:>9}{r['interventions']:>15}"
              f"{r['median_s']:>12.2f}{r['never_reached']:>15.2f}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\ntable -> {args.csv}")


if __name__ == "__main__":
    main()
