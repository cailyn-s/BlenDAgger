"""Statistics and metric loading for the collection / eval metrics, without matplotlib."""
import json
import os
from glob import glob
from typing import Dict, List

import numpy as np
from scipy.stats import t as t_dist

from metrics import task_time


SUMMARY_KEYS = [
    "task_success",
    "task_time_seconds",
    "num_distinct_interventions",
    "num_intervention_steps",
    "mean_intervention_magnitude",
    # mean magnitude over steps where the human commanded motion; skipped when absent
    "mean_intervention_magnitude_moving",
    "total_intervention_magnitude",
    "num_restarts",
]

def ci95_halfwidth(values) -> float:
    """95% CI half-width of the mean: t_{n-1,0.975} * s / sqrt(n)."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        return 0.0
    return float(t_dist.ppf(0.975, n - 1) * arr.std(ddof=1) / np.sqrt(n))


def entry_ci95(entry: dict, context: str = "") -> float:
    """95% CI half-width for a rollout-JSON entry; falls back to z*SEM if no raw values."""
    values = entry.get("values")
    if values:
        return ci95_halfwidth(values)
    print(f"Warning: no 'values' in {context or 'rollout entry'} — "
          f"approximating the 95% CI as 1.96 x the stored SEM.")
    return 1.959964 * float(entry.get("std", 0.0))


def normalize_iterations(entries: List[dict], context: str = "") -> List[dict]:
    """Rollout-JSON iteration entries → {mean, err (95% CI half-width), iteration}."""
    return [
        {
            "mean": float(entry.get("mean", 0.0)),
            "err": entry_ci95(entry, context),
            "iteration": entry.get("iteration", i),
        }
        for i, entry in enumerate(entries)
    ]


def load_all_json_files(folder: str) -> List[dict]:
    json_paths = glob(os.path.join(folder, "*.json"))
    if len(json_paths) == 0:
        return []

    json_paths.sort(key=os.path.getmtime)

    all_data = []
    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)
        # task_time_seconds in meta starts at episode creation, which includes
        # dead time before the operator first squeezes the grip trigger. Recompute
        # it from the grip trigger where the hdf5 logs it (no-op otherwise).
        data = task_time.correct_meta_from_json(path, data)
        all_data.append(data)

    return all_data

def load_meta_file(path: str) -> List[dict]:
    """One round's meta, parsed exactly as load_all_json_files parses a folder's.

    Needed because a round is not always a folder: the real tasks put one session per directory,
    while the sim runs put every round in a single directory.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    data = task_time.correct_meta_from_json(path, data)
    return [data]


def extract_summary_values(
    json_data_list: List[dict],
    summary_keys: List[str],
) -> Dict[str, Dict[str, List[float]]]:
    results = {
        "completed": {key: [] for key in summary_keys}, # completed episodes
        "accumulated": {key: [] for key in summary_keys} # all episodes, including restarts
    }

    warned_keys: set = set()
    for file_data in json_data_list:
        grouped_data = {}

        for demo_key, demo_content in file_data.items():
            is_aborted = bool(demo_content.get("aborted", False))
            summary = demo_content.get("summary", demo_content)
            base_id = demo_key.split("_restart")[0]
            is_restart = "restart" in demo_key

            # "completed" = valid finished episodes → exclude aborted attempts
            # (and restarts). These feed the non-accumulated collection plots.
            if not is_restart and not is_aborted:
                for key in summary_keys:
                    if key not in summary:
                        if key not in warned_keys:
                            print(f"Warning: '{key}' missing from data — skipping (data may predate this field)")
                            warned_keys.add(key)
                        continue
                    results["completed"][key].append(float(summary[key]))

            # "accumulated" = total cost across ALL attempts, INCLUDING aborted
            # ones and restarts (only place aborted episodes are counted).
            if base_id not in grouped_data:
                grouped_data[base_id] = {key: 0.0 for key in summary_keys}

            for key in summary_keys:
                if key not in summary:
                    continue
                val = float(summary[key])
                # Sum everything except success metrics
                if "success" in key:
                    if not is_restart and not is_aborted:
                        grouped_data[base_id][key] = val
                else:
                    grouped_data[base_id][key] += val

        # Transfer grouped sums into the results list
        for base_id in grouped_data:
            for key in summary_keys:
                results["accumulated"][key].append(grouped_data[base_id][key])

    return results

def compute_stats(metric_values: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for key, values in metric_values.items():
        arr = np.array(values, dtype=np.float64)
        # "err" is the 95% CI half-width of the mean, matching the success-rate plots.
        stats[key] = {
            "mean": float(np.mean(arr)) if len(arr) > 0 else 0.0,
            "err": ci95_halfwidth(arr),
        }
    return stats

def compute_stats_per_iteration(
    json_data_list: List[dict],
    summary_keys: List[str],
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    results = {
        "completed": {key: [] for key in summary_keys},
        "accumulated": {key: [] for key in summary_keys}
    }

    for file_data in json_data_list:
        extracted = extract_summary_values([file_data], summary_keys)
        
        stats_completed = compute_stats(extracted["completed"])
        stats_acc = compute_stats(extracted["accumulated"])

        for key in summary_keys:
            results["completed"][key].append(stats_completed[key])
            results["accumulated"][key].append(stats_acc[key])

    return results


def score_episodes(path: str) -> list:
    """The per-episode list from an eval score file, in either shape the robots write it.

    The xArm's `subtask_scores.json` is a bare list of episodes. Other robot stacks write an
    `eval_results.json` that wraps the same episodes in `{task, num_subtasks, summary, episodes:
    [...]}`. The episodes are the same shape in both -- `subtasks: [{index, success, ...}]` -- so
    only the wrapper needs unwrapping.
    """
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.get("episodes", []))


def _max_stage(episode: dict) -> int:
    stage = 0
    for subtask in episode["subtasks"]:
        if subtask["success"]:
            stage += 1
        else:
            break
    return stage


def _load_subtask_fractions(path: str, n_subtasks: int) -> list:
    episodes = score_episodes(path)
    counts = [0] * (n_subtasks + 1)
    for ep in episodes:
        counts[min(_max_stage(ep), n_subtasks)] += 1
    n = len(episodes)
    return [counts[i + 1] / n for i in range(n_subtasks)]


def _n_episodes(path: str) -> int:
    return len(score_episodes(path))


# --- per-episode statistics ---

_EMPTY_POINTS = {"values": np.array([]), "episode": np.array([], dtype=np.int64)}

def _ci95(values: np.ndarray) -> tuple[float, float]:
    """(lo, hi) 95% CI of the mean: mean ± t_{n-1,0.975} * s / sqrt(n)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return (np.nan, np.nan)
    hw = t_dist.ppf(0.975, n - 1) * arr.std(ddof=1) / np.sqrt(n)
    return (float(arr.mean() - hw), float(arr.mean() + hw))

def _episode_means(entry: dict) -> np.ndarray:
    """Collapse {values, episode} to one value per episode.

    The episode is the sample unit for every paper figure: methods differ in how
    many interventions they take, so event-level pooling would weight a method
    that intervenes more heavily than one that intervenes less.
    """
    v = np.asarray(entry["values"], dtype=np.float64)
    e = np.asarray(entry["episode"])
    keep = np.isfinite(v)
    v, e = v[keep], e[keep]
    if v.size == 0:
        return np.array([])
    uniq, inv = np.unique(e, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=uniq.size)
    counts = np.bincount(inv, minlength=uniq.size)
    return sums / counts

def _ci95_cluster(values, episodes, n_boot: int = 2000, seed: int = 0):
    """95% CI of the mean, resampling whole EPISODES.

    Used by the per-intervention figures: each drawn episode contributes its whole
    block of interventions, so within-episode correlation is preserved without
    assuming any correlation structure. The estimate is the intervention-weighted
    mean; only its uncertainty changes.
    """
    v = np.asarray(values, dtype=np.float64)
    e = np.asarray(episodes)
    keep = np.isfinite(v)
    v, e = v[keep], e[keep]
    if v.size < 2:
        return (np.nan, np.nan)
    uniq, inv = np.unique(e, return_inverse=True)
    if uniq.size < 2:
        return _ci95(v)
    sums = np.bincount(inv, weights=v, minlength=uniq.size)
    counts = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, uniq.size, size=(n_boot, uniq.size))
    boot = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (float(lo), float(hi))
