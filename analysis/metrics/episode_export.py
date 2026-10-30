"""Per-episode export: one row per collection episode, for downstream statistics code.

`run_analysis.py` computes its metrics pooled over timesteps; this module reshapes the same
numbers into one row per episode, the unit statistical tests need, and writes them to
`<output_dir>/export/`:

    episodes.csv   one row per collection episode
    rollouts.csv   one row per policy-evaluation rollout
    manifest.json  how the export was produced

Downstream statistics/plotting code reads this format, so the columns are a fixed contract
(see CONTRACT_VERSION).

Two rules govern everything here. `round` and `session` come from the folder the demo was loaded
from and are never reconstructed from an array index — a round that recorded sixteen episodes
would otherwise silently attribute episodes to the wrong condition. And `demo_8_restart1` splits
into `episode: 8, restart: 1` rather than collapsing onto `demo_8`, because different collection
stacks record an abandoned attempt in different shapes.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import subprocess
from datetime import datetime

import numpy as np

# Bump when the required columns change so readers can reject a stale export.
CONTRACT_VERSION = 1

# Identity, then the optional-but-wanted columns; every remaining column is a metric.
IDENTITY = ["dataset", "study", "task", "environment", "method", "round", "session",
            "episode", "restart"]
OPTIONAL = ["participant", "algo", "task_name", "recorded_at", "source_folder",
            "success", "aborted"]

# Copied straight through from each folder's *_meta.json, one row per demo_N key.
META_METRICS = ["task_time_seconds", "num_steps", "num_distinct_interventions",
                "num_intervention_steps", "num_moving_intervention_steps",
                "mean_intervention_magnitude", "mean_intervention_magnitude_moving",
                "total_intervention_magnitude", "num_restarts"]

_DEMO_KEY = re.compile(r"^demo_(\d+)(?:_restart(\d+))?$")
_ROUND = re.compile(r"round(\d+)", re.IGNORECASE)
_STAMP = re.compile(r"(\d{8}_\d{6})")
_ISO_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})")


# ===========================================================================
# Identity parsing
# ===========================================================================

def parse_demo_key(key: str) -> tuple[int, int] | None:
    """(episode, restart) for a `demo_N` / `demo_N_restartM` key, else None.

    Returning None is how non-episode entries are skipped, such as a `_run` block some
    collection stacks write alongside the demos.
    """
    m = _DEMO_KEY.match(str(key))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def round_of(name: str) -> int | None:
    """Collection round from a folder or file name; 0 for the teleop practice folder."""
    if "teleop_practice" in name:
        return 0
    hits = _ROUND.findall(name)
    return int(hits[-1]) if hits else None


def session_round(folder: str, fpath: str | None = None) -> tuple[str, int | None]:
    """(session, round) for one iteration file, taken from the folder it was loaded from.

    The real-robot stacks give a round its own folder, so the folder name carries both. Sim keeps
    every round of a run in one folder and puts the round in the file name instead, so the session
    is qualified with it — one session name per round either way.
    """
    base = os.path.basename(os.path.normpath(folder))
    rnd = round_of(base)
    if rnd is not None:
        return base, rnd
    if fpath:
        rnd = round_of(os.path.basename(fpath))
        if rnd is not None:
            return f"{base}_round{rnd}", rnd
    return base, None


def method_of(algo_label: str, mapping: dict | None = None) -> str:
    """The exported method name, from the run key in the analysis config.

    An ablation keeps its own name (`Blended-noUQ`) and flows through untouched; downstream code
    reports it without needing an entry for it.
    """
    if mapping and algo_label in mapping:
        return str(mapping[algo_label])
    label = algo_label.lower()
    m = re.search(r"_no([A-Za-z0-9]+)", algo_label)
    suffix = f"-no{m.group(1)}" if m else ""
    if "teleop" in label:
        return "Teleop"
    if "blend" in label:
        return f"Blended{suffix}"
    if "dagger" in label:
        return f"HG{suffix}"
    return algo_label


# ===========================================================================
# Provenance: what the collection script recorded about the session
# ===========================================================================

def _stamp_to_iso(text: str) -> str | None:
    for fmt, rx in (("%Y%m%d_%H%M%S", _STAMP), ("%Y-%m-%d-%H:%M:%S", _ISO_STAMP)):
        m = rx.search(text)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt).isoformat()
            except ValueError:
                pass
    return None


def provenance(folder: str, fpath: str | None, meta: dict | None) -> dict:
    """`algo`, `task_name` and `recorded_at` as the collection run recorded them.

    Stacks record it differently: some nest the resolved argument list under a `_run` key in the
    meta, the xArm writes a `collect_config_<stamp>.yaml` beside the data, and sim writes a
    `run_config.yaml`. The recording timestamp is in the data file's name in every case.
    """
    out = {"algo": None, "task_name": None, "recorded_at": None}
    run = (meta or {}).get("_run")
    if isinstance(run, dict):
        args = run.get("resolved_args") or {}
        out["algo"] = args.get("algo")
        out["task_name"] = args.get("task_name")
        out["recorded_at"] = run.get("timestamp")

    if out["algo"] is None:
        cfgs = sorted(glob.glob(os.path.join(folder, "collect_config_*.yaml")) +
                      glob.glob(os.path.join(folder, "run_config.yaml")))
        if cfgs:
            out.update(_scan_yaml(cfgs[-1], ("algo", "task_name")))

    if not out["recorded_at"] and fpath:
        out["recorded_at"] = _stamp_to_iso(os.path.basename(fpath))
    # Sim records the run under `run_name` and `algo` and never writes a `task_name`; its folder
    # name is what the run was called. Falling back to it keeps downstream code's
    # folder-vs-recorded-name check defined instead of comparing against a blank. The real-robot
    # stacks always record one, so this never fires there and cannot mask a genuine mismatch.
    if out["task_name"] is None:
        out["task_name"] = os.path.basename(os.path.normpath(folder))
    return {k: v for k, v in out.items() if v is not None}


def _scan_yaml(path: str, keys: tuple[str, ...]) -> dict:
    """Top-level `key: value` lines, read without pulling in a YAML parser."""
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                for key in keys:
                    if line.startswith(f"{key}:"):
                        val = line.split(":", 1)[1].strip().strip("'\"")
                        if val and val != "null":
                            out[key] = val
    except OSError:
        pass
    return out


# ===========================================================================
# The registry: identity for every episode, filled in as the passes run
# ===========================================================================

class EpisodeRegistry:
    """Every episode seen this run, keyed on the folder it came from and its demo key.

    The episode id this hands out is what the `__episode` companion arrays in distributions.npz
    carry, so the collection pass over the full files and the training pass over the proc files
    land on the same row — they read different files and would otherwise number independently.
    """

    def __init__(self, dataset: str, study: str, task: str, environment: str):
        self.base = dict(dataset=dataset, study=study, task=task, environment=environment)
        self._ids: dict[tuple[str, str], int] = {}
        self.rows: dict[int, dict] = {}

    def id_of(self, folder: str, demo: str, method: str, fpath: str | None = None) -> int:
        """The id for one episode, creating its row the first time it is seen.

        Keyed on the SESSION rather than the folder: sim keeps every round of a run in one folder
        and restarts its demo numbering each round, so a folder-level key would make round 2's
        `demo_0` collide with round 1's and silently overwrite it.
        """
        session, _ = session_round(folder, fpath)
        key = (os.path.abspath(folder), session, demo)
        if key in self._ids:
            return self._ids[key]
        ep_id = len(self._ids)
        self._ids[key] = ep_id
        _, rnd = session_round(folder, fpath)
        parsed = parse_demo_key(demo)
        self.rows[ep_id] = dict(
            self.base, method=method, round=rnd, session=session,
            episode=parsed[0] if parsed else None,
            restart=parsed[1] if parsed else 0,
            source_folder=os.path.abspath(folder),
        )
        return ep_id

    def row(self, folder: str, demo: str, method: str, fpath: str | None = None) -> dict:
        return self.rows[self.id_of(folder, demo, method, fpath)]

    def update(self, ep_id: int, values: dict) -> None:
        self.rows[ep_id].update({k: v for k, v in values.items() if v is not None})


# ===========================================================================
# Per-episode reductions of the pooled arrays
# ===========================================================================

def episode_means(entry: dict) -> dict[int, float]:
    """{episode id: mean of that episode's values} for one `{values, episode}` pooled entry."""
    v = np.asarray(entry.get("values", []), dtype=np.float64).ravel()
    e = np.asarray(entry.get("episode", [])).ravel()
    if v.size == 0 or v.size != e.size:
        return {}
    keep = np.isfinite(v)
    v, e = v[keep], e[keep]
    if v.size == 0:
        return {}
    uniq, inv = np.unique(e, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=uniq.size)
    counts = np.bincount(inv, minlength=uniq.size)
    return {int(k): float(s / c) for k, s, c in zip(uniq, sums, counts)}


def episode_counts(entry: dict) -> dict[int, int]:
    """{episode id: how many values it contributed} — the group count, not a mean."""
    e = np.asarray(entry.get("episode", [])).ravel()
    v = np.asarray(entry.get("values", []), dtype=np.float64).ravel()
    if e.size == 0 or e.size != v.size:
        return {}
    uniq, counts = np.unique(e[np.isfinite(v)], return_counts=True)
    return {int(k): int(c) for k, c in zip(uniq, counts)}


# ===========================================================================
# Collection metrics, straight out of meta.json
# ===========================================================================

def meta_rows(meta: dict) -> dict[str, dict]:
    """{demo key: flat summary} for every episode record in a *_meta.json.

    Sim nests the numbers under `summary` and the real-robot metas are flat; a `_run` block, if
    present, is not an episode and `parse_demo_key` keeps it out.
    """
    out = {}
    for key, value in meta.items():
        if not isinstance(value, dict) or parse_demo_key(key) is None:
            continue
        summary = value.get("summary", value)
        row = dict(summary)
        row["aborted"] = bool(value.get("aborted", summary.get("aborted", False)))
        row["task_success"] = bool(summary.get("task_success", False))
        out[key] = row
    return out


def canonical_names(hdf5_demos: list[str], meta_keys: list[str]) -> dict[str, str]:
    """{hdf5 demo name: the meta.json key that names the same episode}.

    The real-robot stacks use the same names in both. Sim does not: its hdf5 keeps the source
    dataset's demo indices (demo_12, demo_17, ...) while its meta numbers episodes from zero, so
    keying rows on the hdf5 name leaves the collection metrics and the OOD metrics on separate
    rows. When the names do not already agree, the only correspondence available is order — both
    files are written per episode as collection proceeds — so pair them positionally.
    """
    meta_episodes = sorted((k for k in meta_keys if (parse_demo_key(k) or (0, 1))[1] == 0),
                           key=lambda k: parse_demo_key(k)[0])
    demos = sorted(hdf5_demos, key=lambda k: (parse_demo_key(k) or (0, 0)))
    if set(demos) <= set(meta_episodes):
        return {d: d for d in demos}
    if len(demos) != len(meta_episodes):
        return {d: d for d in demos}
    return dict(zip(demos, meta_episodes))


def collection_values(row: dict) -> dict:
    """The meta's per-episode numbers, under the exported column names."""
    out = {k: row[k] for k in META_METRICS if k in row}
    out["success"] = bool(row.get("task_success", False))
    out["aborted"] = bool(row.get("aborted", False))
    return out


# ===========================================================================
# Arbitration: mean blend weight per episode
# ===========================================================================

def alpha_during_intervention(alphas, intervention_mask) -> tuple[float | None, list]:
    """(mean over DoFs, per-DoF means) of alpha over an episode's intervened steps.

    Blank rather than zero when the episode has no intervention or the run logged no alphas: zero
    means the arbitration handed everything to the human, which is a different statement.
    """
    if alphas is None:
        return None, []
    a = np.asarray(alphas, dtype=np.float64)
    iv = np.asarray(intervention_mask, dtype=bool)
    if a.ndim == 1:
        a = a[:, None]
    n = min(len(a), len(iv))
    a, iv = a[:n], iv[:n]
    if not iv.any():
        return None, []
    during = a[iv]
    per_dof = [float(x) for x in during.mean(axis=0)]
    return float(during.mean()), per_dof


# ===========================================================================
# rollouts.csv
# ===========================================================================

ROLLOUT_COLUMNS = ["dataset", "study", "task", "environment", "method", "round", "rollout",
                   "success", "checkpoint", "session", "participant"]


def _rollouts_from_subtask_scores(path: str) -> list[dict]:
    """One row per rollout from an xArm `subtask_scores.json`; success = every subtask passed."""
    with open(path) as fh:
        episodes = json.load(fh)
    rows = []
    for i, ep in enumerate(episodes):
        subtasks = ep.get("subtasks") or []
        ok = bool(subtasks) and all(bool(s.get("success")) for s in subtasks)
        rows.append(dict(rollout=int(ep.get("episode", i + 1)), success=int(ok)))
    return rows


def _rollouts_from_eval_results(path: str) -> list[dict]:
    """One row per rollout from an `eval_results.json` (other robot stacks), which records `all_success`."""
    with open(path) as fh:
        payload = json.load(fh)
    rows = []
    for i, ep in enumerate(payload.get("episodes", [])):
        subtasks = ep.get("subtasks") or []
        ok = bool(ep["all_success"]) if "all_success" in ep else (
            bool(subtasks) and all(bool(s.get("success")) for s in subtasks))
        rows.append(dict(rollout=int(ep.get("episode", i + 1)), success=int(ok)))
    return rows


_EPOCH = re.compile(r"epoch=?(\d+)")


def checkpoint_of(folder: str) -> str | None:
    """The checkpoint an eval folder ran, read off the `ckpt_dir` its config recorded.

    Round alone cannot separate two evaluations of the same round at different epochs, which is
    what the checkpoint column is for.
    """
    for name in ("args.yaml", "collect_config_*.yaml"):
        for path in sorted(glob.glob(os.path.join(folder, name))):
            for line in open(path, errors="ignore"):
                if line.startswith("ckpt_dir:"):
                    m = _EPOCH.search(line)
                    if m:
                        return f"epoch{int(m.group(1))}"
    return None


def _stamp_checkpoint(rows: list[dict], folder: str) -> None:
    """Tag rows with the checkpoint the folder's config recorded, if it recorded one."""
    ckpt = checkpoint_of(folder)
    if ckpt:
        for row in rows:
            row.setdefault("checkpoint", ckpt)


def rollouts_from_folder(folder: str) -> list[dict]:
    """Rollout rows from an `eval_*` folder, whichever of the two result files it holds."""
    subtask = os.path.join(folder, "subtask_scores.json")
    if os.path.exists(subtask):
        rows = _rollouts_from_subtask_scores(subtask)
    else:
        results = os.path.join(folder, "eval_results.json")
        if not os.path.exists(results):
            return []
        rows = _rollouts_from_eval_results(results)
    _stamp_checkpoint(rows, folder)
    return rows


def rollouts_from_realmatch(path: str) -> list[dict]:
    """Rollout rows from a sim `eval_composite_results/*.json`, one row per element of `success`.

    Every checkpoint the run evaluated is kept: `epochs.<epoch>.success` is already a per-rollout
    list, and which checkpoint to report is left to downstream code.
    """
    with open(path) as fh:
        payload = json.load(fh)
    rows = []
    for epoch, entry in (payload.get("epochs") or {}).items():
        for i, value in enumerate(entry.get("success", [])):
            rows.append(dict(rollout=i + 1, success=int(bool(value)), checkpoint=str(epoch)))
    return rows


def build_rollouts(spec: dict, base: dict) -> list[dict]:
    """Every rollout row for one analysis run.

    `spec` maps a method name to `{round: path}`, where the path is an `eval_*` folder or a sim
    result JSON. Round is which round's data trained the policy that was evaluated, so 0 is the
    pretrained base policy — the same evaluation is reported under both methods, because both
    started from it.
    """
    rows = []
    for method, rounds in (spec or {}).items():
        for rnd, path in (rounds or {}).items():
            path = str(path)
            name = os.path.basename(path)
            if os.path.isdir(path):
                found = rollouts_from_folder(path)
            elif name == "subtask_scores.json" and os.path.isfile(path):
                found = _rollouts_from_subtask_scores(path)
                _stamp_checkpoint(found, os.path.dirname(path))
            elif name == "eval_results.json" and os.path.isfile(path):
                found = _rollouts_from_eval_results(path)
                _stamp_checkpoint(found, os.path.dirname(path))
            elif os.path.isfile(path):
                found = rollouts_from_realmatch(path)
            else:
                print(f"  [export] rollouts: {path} not found — skipping")
                continue
            if not found:
                print(f"  [export] rollouts: no episodes in {path} — skipping")
                continue
            session = os.path.basename(os.path.normpath(
                os.path.dirname(path) if os.path.isfile(path) else path))
            for row in found:
                rows.append(dict(base, method=method, round=int(rnd),
                                 session=session, participant="", **row))
    return rows


# ===========================================================================
# Writing
# ===========================================================================

def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _order(columns: set) -> list[str]:
    """Identity first, then the optional columns, then every metric alphabetically."""
    head = [c for c in IDENTITY + OPTIONAL if c in columns]
    return head + sorted(columns - set(head))


def _cell(value):
    """Blank for anything undefined — a metric that does not apply is not a zero."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{float(value):.10g}"
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_csv(path: str, rows: list[dict], columns: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = columns or _order({c for r in rows for c in r})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in columns})


def _row_key(row) -> tuple:
    """What identifies one episode across runs: the session, its index, and the restart.

    Written out rather than `or ""`, which silently loses episode 0 for being falsy.
    """
    def cell(value, default=""):
        return default if value is None or value == "" else str(value)
    return (cell(row.get("session")), cell(row.get("episode")), cell(row.get("restart"), "0"))


def carry_forward(path: str, rows: list[dict]) -> int:
    """Fill each row's empty columns from the previous export, and keep rows this run never saw.

    A partial recompute (`--only-metrics`) computes some metric families and not others. Without
    this it would blank the families it skipped, which is worse than the run not happening.
    """
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as fh:
        previous = {_row_key(r): r for r in csv.DictReader(fh)}
    filled = 0
    for row in rows:
        old = previous.pop(_row_key(row), None)
        if not old:
            continue
        for column, value in old.items():
            if value not in (None, "") and row.get(column) in (None, ""):
                row[column] = value
                filled += 1
    # A session this run did not touch at all keeps its row rather than vanishing from the table.
    rows.extend(previous.values())
    return filled


def write_export(output_dir: str, registry: EpisodeRegistry, rollouts: list[dict],
                 manifest: dict, partial: bool = False) -> str:
    """Write episodes.csv, rollouts.csv and manifest.json, and return the export directory."""
    export_dir = os.path.join(output_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    rows = [registry.rows[i] for i in sorted(registry.rows)]
    rows.sort(key=lambda r: (str(r.get("method")), r.get("round") if r.get("round") is not None
                             else -1, str(r.get("session")),
                             r.get("episode") if r.get("episode") is not None else -1,
                             r.get("restart") or 0))
    for row in rows:
        row.setdefault("participant", "")

    episodes_path = os.path.join(export_dir, "episodes.csv")
    if partial:
        filled = carry_forward(episodes_path, rows)
        rows.sort(key=lambda r: (str(r.get("method")), str(r.get("session")),
                                 int(r["episode"]) if str(r.get("episode") or "").isdigit() else -1))
        print(f"  [partial] kept {filled} value(s) from the previous export "
              f"for the metric families this run did not recompute")
    write_csv(episodes_path, rows)
    print(f"  Episodes           → {episodes_path}  ({len(rows)} rows)")

    rollouts_path = os.path.join(export_dir, "rollouts.csv")
    if rollouts:
        write_csv(rollouts_path, rollouts, ROLLOUT_COLUMNS)
        print(f"  Rollouts           → {rollouts_path}  ({len(rollouts)} rows)")
    elif os.path.exists(rollouts_path):
        os.remove(rollouts_path)

    manifest = dict(manifest)
    manifest.update(contract_version=CONTRACT_VERSION,
                    generated_at=datetime.now().isoformat(timespec="seconds"),
                    code_version=_git_head(),
                    n_episodes=len(rows), n_rollouts=len(rollouts),
                    # True when some metric families were carried forward rather than computed.
                    partial=bool(partial))
    manifest_path = os.path.join(export_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=_json_safe)
    print(f"  Export manifest    → {manifest_path}")
    return export_dir


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ===========================================================================
# Config
# ===========================================================================

# Which analysis run maps to which task and side of the study, when the config does not say.
_TASK_HINTS = (("cupboard", "cupboard"), ("almond", "almond"), ("snack", "snack"),
               ("lemon", "lemon"), ("microwave", "microwave"))


def export_settings(cfg, run_name: str, output_dir: str) -> dict:
    """Fill in `dataset`, `study`, `task` and `environment` from the config, or infer them."""
    block = cfg.get("export", None) if hasattr(cfg, "get") else None
    block = dict(block) if block else {}
    dataset = str(block.get("dataset") or os.path.basename(os.path.normpath(output_dir))
                  or run_name)
    task = block.get("task")
    if not task:
        for token, name in _TASK_HINTS:
            if token in dataset.lower() or token in run_name.lower():
                task = name
                break
    study = block.get("study") or ("user_study" if "user_study" in dataset.lower() else "expert")
    environment = block.get("environment") or (
        "sim" if any(t in dataset.lower() for t in ("sim", "realmatch")) else "real")
    return dict(dataset=dataset, study=str(study), task=str(task or run_name),
                environment=str(environment),
                methods=dict(block.get("methods") or {}),
                rollouts={k: dict(v) for k, v in (block.get("rollouts") or {}).items()},
                enabled=bool(block.get("enabled", True)))
