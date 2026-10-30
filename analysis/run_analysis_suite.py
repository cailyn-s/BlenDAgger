#!/usr/bin/env python3
"""
Run the FULL analysis suite for one task, in a single command:

    python analysis/run_analysis_suite.py --config almond

Reads conf/<config>.yaml (a tiny per-task config) and, from the `task`,
`data_root`, `algos`, `num_episodes`, `eval_num_episodes` fields, it:

  1. AUTODETECTS round folders under <data_root>/<task>/ :
        <task>_<algo>_round<N>          (collection; each round = one iteration)
        <task>_<algo>_no<X>_round<N>    (ablations, only with --ablations / include_ablations)
        eval_<algo>_round<N>            (evaluation; subtask-success)
        eval_pretrain, <task>_pretrain
     Rounds are episode-count-validated (demos in *_full.hdf5 vs num_episodes;
     eval episodes in subtask_scores.json vs eval_num_episodes). A folder that
     exists but is short on data is WARNED about and skipped; the suite still
     runs on whatever rounds are complete, so dagger/blended need not have the
     same number of rounds.

  2. Generates a temp Hydra config and runs:
        run_analysis.py  → ood/, timing/, action_change/  (proc_* files)
                         → export/  one row per episode + one per eval rollout, for
                           downstream statistics/plotting code

  3. Writes alpha/sessions.json and eval/sessions.json, session manifests for
     downstream plotting (rounds whose alphas are constant are warned + skipped).

Run it with the Python environment used for the individual scripts; sub-steps
inherit this interpreter.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import h5py
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics.config import load_task_config  # noqa: E402
from metrics import episode_export
from metrics import task_time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONF = os.path.join(REPO, "conf")
PY = sys.executable
# Only the computation stage runs here; figures are drawn by downstream plotting code.
STEPS = ("analysis",)
# Ablation colors, chosen to follow HG-DAgger (lightblue) and BlenDAgger (#d8b4f8).
ABLATION_COLORS = ["#fc9292", "#fc92ee", "green", "yellow", "orange", "blue", "teal"]
# fallback only -- the live values come from conf/shared/algo_colors.yaml

C_WARN = "\033[33m"
C_HEAD = "\033[1m"
C_END = "\033[0m"


def warn(msg):
    print(f"  {C_WARN}[warn]{C_END} {msg}")


def run(cmd):
    print(f"\n{C_HEAD}$ " + " ".join(str(c) for c in cmd) + C_END, flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


# --------------------------------------------------------------------------- #
# Processed training files
# --------------------------------------------------------------------------- #
# The states the policy is fit on. OOD, trajectory similarity and the training-file action
# deltas are measured against these; without one, a round is silently dropped from those
# metrics while still contributing collection metrics.
#
# The flags are the project's data-processing contract and are not options: eef/rel matches the
# 7-D executed commands the collection writes, --trim-zeros drops stationary frames, --step-grip
# and --resize-images match how the training sets were built. A proc file made with different
# flags is not comparable with the ones already on disk.
PROC_FLAGS = ["--act-space", "eef", "--act-type", "rel",
              "--trim-zeros", "--resize-images", "--step-grip"]
# Processing script from the companion xArm repository (this repo is a submodule of it by default).
PROC_SCRIPT = os.environ.get("BLENDAGGER_PROC_SCRIPT",
                             os.path.join(os.path.dirname(REPO), "scripts", "data_processing.py"))
MAKE_PROC = True          # --no-make-proc turns this off


def ensure_proc(folder, label):
    """Make the processed file for one round folder if it is missing.

    Returns True when the folder has one afterwards. Generation is skipped, with a warning
    rather than an error, when the source file or the processing script is absent — a round
    without a proc file still contributes its collection metrics.
    """
    if glob.glob(os.path.join(folder, "proc_*.hdf5")):
        return True
    full = _round_full_file(folder)
    if full is None:
        warn(f"{label}: no *_full.hdf5, so no processed file can be made")
        return False
    if not MAKE_PROC:
        warn(f"{label}: no proc_*.hdf5 and --no-make-proc is set — OOD/DTW/action-change "
             "will skip this round")
        return False
    if not os.path.exists(PROC_SCRIPT):
        warn(f"{label}: no proc_*.hdf5 and {PROC_SCRIPT} is missing — cannot make one "
             "(the script lives in the companion xArm repository; point BLENDAGGER_PROC_SCRIPT at it)")
        return False
    out = os.path.join(folder, "proc_" + os.path.basename(full))
    print(f"  [proc] {label}: building {os.path.basename(out)} ...", end=" ", flush=True)
    done = subprocess.run([PY, PROC_SCRIPT, "--input", full, "--output", out] + PROC_FLAGS,
                          capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(out):
        print("failed.")
        warn(f"{label}: data_processing.py failed; "
             f"{(done.stderr or '').strip().splitlines()[-1] if done.stderr else 'no output'}")
        return False
    print("done.")
    return True


# --------------------------------------------------------------------------- #
# Detection + validation
# --------------------------------------------------------------------------- #
def _round_full_file(folder):
    """The non-proc *_full.hdf5 in a round folder (or None)."""
    fs = [f for f in glob.glob(os.path.join(folder, "*_full.hdf5"))
          if not os.path.basename(f).startswith("proc_")]
    return sorted(fs)[0] if fs else None


def _count_demos(hdf5_path):
    """Number of VALID (non-aborted) episodes a round ran.

    Aborted episodes are recorded in the hdf5 but flagged `aborted: true` in the
    sibling *_meta.json (and the collection metrics already skip them), so they
    must not count toward the episode-completeness check.

    Counted from the meta.json where there is one, because that is the record of
    what the round *ran*, while the full file is the record of what was *saved*.
    They differ when an episode is discarded at the prompt: the trajectory is
    gone but the episode happened, and its collection metrics are still in the
    meta.
    """
    with h5py.File(hdf5_path, "r") as f:
        saved = len(f["data"].keys())
    meta_path = task_time.meta_path_for(hdf5_path)
    if not meta_path:
        return saved
    with open(meta_path) as fh:
        meta = json.load(fh)
    episodes = [v for k, v in meta.items()
                if isinstance(v, dict) and episode_export.parse_demo_key(k) is not None]
    n_aborted = sum(1 for v in episodes if v.get("aborted", False))
    return len(episodes) - n_aborted


def detect_rounds(task_dir, task, algo_key, required):
    """[(round_n, folder, full_file)] passing the episode check, round-sorted."""
    pat = re.compile(rf"^{re.escape(task)}_{re.escape(algo_key)}_round(\d+)$")
    found = []
    for name in sorted(os.listdir(task_dir)):
        m = pat.match(name)
        if not m:
            continue
        folder = os.path.join(task_dir, name)
        if not os.path.isdir(folder):
            continue
        n = int(m.group(1))
        full = _round_full_file(folder)
        if full is None:
            warn(f"{name}: no *_full.hdf5 found — skipping")
            continue
        cnt = _count_demos(full)
        if cnt < required:
            warn(f"{name}: only {cnt}/{required} episodes — incomplete, skipping")
            continue
        if cnt != required:
            warn(f"{name}: {cnt} episodes (expected {required}) — including anyway")
        if not ensure_proc(folder, name):
            warn(f"{name}: no proc_*.hdf5 — OOD/DTW/action-change will skip this "
                 f"round (collection/alpha still use it)")
        found.append((n, folder, full))
    found.sort(key=lambda t: t[0])
    return found


def explicit_rounds(task_dir, folder_names, required, what):
    """[(round_n, folder, full_file)] for a config-pinned list of round folders.

    Round numbers come from list position, so a folder-name suffix identifying
    a training lineage never reaches a plot label. Missing/incomplete folders
    are warned about and dropped.
    """
    found = []
    for i, name in enumerate(folder_names):
        folder = name if os.path.isabs(name) else os.path.join(task_dir, name)
        if not os.path.isdir(folder):
            warn(f"{what}: {name} not found — skipping")
            continue
        full = _round_full_file(folder)
        if full is None:
            warn(f"{what}: {name} has no *_full.hdf5 — skipping")
            continue
        cnt = _count_demos(full)
        if cnt < required:
            warn(f"{what}: {name} has only {cnt}/{required} episodes — skipping")
            continue
        if cnt != required:
            warn(f"{what}: {name} has {cnt} episodes (expected {required}) — including anyway")
        if not ensure_proc(folder, f"{what}: {name}"):
            warn(f"{what}: {name} has no proc_*.hdf5 — OOD/DTW/action-change will "
                 f"skip this round (collection/alpha still use it)")
        found.append((i + 1, folder, full))
    return found


def pattern_rounds(task_dir, pattern, required, what, exclude=()):
    """[(round_n, folder, full_file)] for folders matching a config-supplied regex.

    The user study names its folders after the participant rather than the task
    (`<participant>_dagger_round1`), so the strict <task>_<algo>_round<N> regex cannot find them. The
    pattern captures the round in its last group; folders are ordered by name so a participant's
    two policies stay adjacent.
    """
    pat = re.compile(pattern)
    found = []
    for name in sorted(os.listdir(task_dir)):
        m = pat.match(name)
        if not m or any(name.startswith(f"{e}_") or name == e for e in exclude):
            continue
        folder = os.path.join(task_dir, name)
        if not os.path.isdir(folder):
            continue
        # An empty last group means the name carries no round number — the teleop practice run,
        # which happens before round 1 and under no policy, so it is round 0.
        n = int(m.groups()[-1]) if (m.groups()[-1] or "").isdigit() else 0
        full = _round_full_file(folder)
        if full is None:
            warn(f"{what}: {name} has no *_full.hdf5 — skipping")
            continue
        cnt = _count_demos(full)
        if cnt < required:
            warn(f"{what}: {name} has only {cnt}/{required} episodes — skipping")
            continue
        if not ensure_proc(folder, f"{what}: {name}"):
            warn(f"{what}: {name} has no proc_*.hdf5 — OOD/DTW/action-change will "
                 f"skip it (collection/alpha still use it)")
        found.append((n, folder, full))
    return found


def detect_ablation_rounds(task_dir, task, algo_key, required):
    """{ablation_suffix: [(round_n, folder, full_file)]} for
    <task>_<algo>_no<Suffix>_round<N> folders, each passing the episode check."""
    pat = re.compile(rf"^{re.escape(task)}_{re.escape(algo_key)}_no([A-Za-z0-9]+)_round(\d+)$")
    groups = {}
    for name in sorted(os.listdir(task_dir)):
        m = pat.match(name)
        if not m:
            continue
        suffix, n = m.group(1), int(m.group(2))
        folder = os.path.join(task_dir, name)
        full = _round_full_file(folder)
        if full is None:
            warn(f"{name}: no *_full.hdf5 found — skipping")
            continue
        cnt = _count_demos(full)
        if cnt < required:
            warn(f"{name}: only {cnt}/{required} episodes — incomplete, skipping")
            continue
        if cnt != required:
            warn(f"{name}: {cnt} episodes (expected {required}) — including anyway")
        if not ensure_proc(folder, name):
            warn(f"{name}: no proc_*.hdf5 — OOD/DTW/action-change will skip this "
                 f"round (collection/alpha still use it)")
        groups.setdefault(suffix, []).append((n, folder, full))
    for s in groups:
        groups[s].sort(key=lambda t: t[0])
    return groups


def _eval_results_file(folder):
    """The eval result file in a folder: subtask_scores.json (xArm) or eval_results.json."""
    for name in ("subtask_scores.json", "eval_results.json"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def _n_eval_episodes(path):
    payload = json.load(open(path))
    return len(payload if isinstance(payload, list) else payload.get("episodes", []))


def detect_eval_rounds(task_dir, algo_key, required):
    """[(round_n, subtask_scores_path)] for eval_<algo>_round<N>, round-sorted.

    Under-`required` evals are INCLUDED (not skipped); the eval figures annotate
    such bars with their rollout count. Only truly empty / missing evals are dropped.
    """
    pat = re.compile(rf"^eval_{re.escape(algo_key)}_round(\d+)$")
    out = []
    for name in sorted(os.listdir(task_dir)):
        m = pat.match(name)
        if not m:
            continue
        s = _eval_results_file(os.path.join(task_dir, name))
        if s is None:
            warn(f"{name}: no subtask_scores.json / eval_results.json — skipping eval")
            continue
        n_ep = _n_eval_episodes(s)
        if n_ep == 0:
            warn(f"{name}: 0 eval episodes — skipping")
            continue
        if n_ep < required:
            warn(f"{name}: only {n_ep}/{required} eval episodes — including (annotated)")
        out.append((int(m.group(1)), s))
    out.sort(key=lambda t: t[0])
    return out


def resolve_extra_eval(task_dir, spec, required):
    """Resolve a --extra-eval spec to [label, subtask_scores_path] (or None).

    spec = "[LABEL=]FOLDER". FOLDER is a folder under task_dir (or an absolute
    path, or a direct subtask_scores.json). LABEL defaults to the folder name
    with the leading 'eval_' stripped; "\\n" in a label is kept literal for
    the eval figures to turn into a grouping sub-label.
    """
    label, folder = (spec.split("=", 1) if "=" in spec else (None, spec))
    folder = folder.strip()

    cand = folder
    if not os.path.isabs(cand) and not os.path.exists(cand):
        cand = os.path.join(task_dir, folder)          # try under the task dir
    s = cand if cand.endswith(".json") else os.path.join(cand, "subtask_scores.json")
    if not os.path.exists(s):
        warn(f"extra-eval '{spec}': subtask_scores.json not found ({s}) — skipping")
        return None

    n_ep = len(json.load(open(s)))
    if n_ep == 0:
        warn(f"extra-eval '{spec}': 0 episodes — skipping")
        return None
    if label is None:
        base = os.path.basename(folder.rstrip("/")) or os.path.basename(cand.rstrip("/"))
        label = base[len("eval_"):] if base.startswith("eval_") else base
    if n_ep < required:
        warn(f"extra-eval {label.replace(chr(10),' ')}: only {n_ep}/{required} "
             f"episodes — including (annotated)")
    else:
        print(f"  extra-eval {label.replace(chr(10),' '):20s}: {n_ep} episodes")
    return [label, s]


def detect_eval_pretrain(task_dir, required):
    s = _eval_results_file(os.path.join(task_dir, "eval_pretrain"))
    if s is None:
        warn("eval_pretrain: no eval result file — pretrain eval omitted")
        return None
    n_ep = _n_eval_episodes(s)
    if n_ep == 0:
        warn("eval_pretrain: 0 episodes — omitting")
        return None
    if n_ep < required:
        warn(f"eval_pretrain: only {n_ep}/{required} episodes — including (annotated)")
    return s


from metrics.alpha import alpha_is_constant as _alpha_is_constant  # noqa: E402
from metrics.alpha import has_blend_fields as _has_blend_fields    # noqa: E402


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def algo_color(cfg, algos, algo):
    """Colors are presentation: they live in the task's plot.yaml as algo_colors."""
    cmap = (cfg.get("algo_colors") or {}) if hasattr(cfg, "get") else {}
    return cmap.get(algo) or algos[algo].get("color")


def detect_runs(task_dir, task, algos, req, include_ablations, cfg, quiet=False):
    """Every run the suite plots, in canonical order.

    Returns (runs, run_folders, present, labels, colors, alpha_specs).
    This is the single definition of the ablation naming rule
    ("<base> (no <suffix>)") and of the colour order (ABLATION_COLORS in
    detection order), so nothing downstream reimplements either -- see
    plot_maps_for() for the presentation-only view of the same thing.
    """
    # ---- collection rounds per algo (+ optional ablations) ----
    runs, run_folders, alpha_specs = {}, {}, []
    present, labels, colors = [], [], []
    abl_i = 0
    # Optional cfg.rounds pins the exact collection folders per algo (a training
    # lineage whose names don't match the strict <task>_<algo>_round<N> regex).
    _pinned_cfg = cfg.get("rounds", None)
    pinned = OmegaConf.to_container(_pinned_cfg, resolve=True) if _pinned_cfg else {}
    # cfg.rounds_pattern matches folders by regex instead, for a layout the strict
    # <task>_<algo>_round<N> convention does not cover (the user study's per-participant folders).
    _pat_cfg = cfg.get("rounds_pattern", None)
    patterns = OmegaConf.to_container(_pat_cfg, resolve=True) if _pat_cfg else {}
    excluded = list(cfg.get("exclude_sessions", []) or [])
    for algo in list(algos.keys()):
        if algo in pinned:
            rounds = explicit_rounds(task_dir, pinned[algo], req, f"rounds.{algo}")
        elif algo in patterns:
            rounds = pattern_rounds(task_dir, patterns[algo], req,
                                    f"rounds_pattern.{algo}", excluded)
        else:
            rounds = detect_rounds(task_dir, task, algo, req)
        if rounds:
            folders = [fol for _, fol, _ in rounds]
            runs[algo] = folders
            run_folders[algo] = folders
            present.append(algo)
            labels.append(algos[algo]["label"])
            colors.append(algo_color(cfg, algos, algo))
            if not quiet:
                print(f"  {algos[algo]['label']:20s}: rounds "
                  f"{[n for n, _, _ in rounds]} ({len(folders)} iterations)")
            for n, fol, full in rounds:
                alpha_specs.append(dict(algo=algo, n=n, folder=fol, full=full,
                                        label=f"{algos[algo]['label']} round {n}",
                                        color=algo_color(cfg, algos, algo)))
        else:
            warn(f"algo '{algo}': no complete rounds — excluded from analysis")

        if include_ablations:
            base = algos[algo]["label"].replace(" (Ours)", "").strip()
            for suffix, abl_rounds in sorted(detect_ablation_rounds(
                    task_dir, task, algo, req).items()):
                key = f"{algo}_no{suffix}"
                _lbls = cfg.get("ablation_labels") or {}
                lbl = f"{base} ({_lbls.get(suffix) or f'no {suffix}'})"
                _abl = cfg.get("ablation_colors") or {}
                col = _abl.get(suffix) or ABLATION_COLORS[abl_i % len(ABLATION_COLORS)]
                abl_i += 1
                folders = [fol for _, fol, _ in abl_rounds]
                runs[key] = folders
                run_folders[key] = folders
                present.append(key)
                labels.append(lbl)
                colors.append(col)
                if not quiet:
                    print(f"  {lbl:20s}: rounds "
                      f"{[n for n, _, _ in abl_rounds]} ({len(folders)} iterations)")
                for n, fol, full in abl_rounds:
                    alpha_specs.append(dict(algo=key, n=n, folder=fol, full=full,
                                            label=f"{lbl} round {n}", color=col))
    return runs, run_folders, present, labels, colors, alpha_specs


def plot_maps_for(cfg, quiet=True):
    """(names, labels, colors) for a config, ablations included.

    The presentation-only view of detect_runs, for callers that need the
    suite's naming and colour order without re-deriving it, such as
    downstream plotting code.
    """
    task = cfg.task
    task_dir = str(cfg.get("data_dir", None) or os.path.join(cfg.data_root, task))
    algos = OmegaConf.to_container(cfg.algos, resolve=True)
    include_ablations = bool(cfg.get("include_ablations", False))
    out = detect_runs(task_dir, task, algos, int(cfg.num_episodes),
                      include_ablations, cfg, quiet=quiet)
    _, _, names, labels, colors, _ = out
    return names, labels, colors


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="task config in conf/analysis/real/ (without .yaml), e.g. almond")
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS,
                    help="steps to skip (only 'analysis' exists; accepted for "
                         "compatibility, no figures are drawn here)")
    ap.add_argument("--only", nargs="*", default=[], choices=STEPS,
                    help="run only these steps (only 'analysis' exists; accepted for "
                         "compatibility, no figures are drawn here)")
    ap.add_argument("--ablations", action="store_true",
                    help="also include <task>_<algo>_no*_round<N> ablation folders")
    ap.add_argument("--no-ablations", action="store_true",
                    help="exclude ablations (overrides config include_ablations)")
    ap.add_argument("--no-make-proc", action="store_true",
                    help="do not build a missing proc_*.hdf5; skip that round's "
                         "OOD/DTW/action-change instead")
    ap.add_argument("--only-metrics", nargs="*", default=[],
                    choices=["ood", "trajsim", "collection"],
                    help="recompute only these metric families; the rest are "
                         "carried over from the existing distributions.npz")
    ap.add_argument("--no-titles", action="store_true",
                    help="accepted for compatibility; recorded in the generated "
                         "config but no figures are drawn here")
    ap.add_argument("--plot-only", action="store_true",
                    help="accepted for compatibility; skips recomputation and draws "
                         "no figures here (plotting reads the saved distributions.npz)")
    ap.add_argument("--extra-eval", action="append", nargs="+", default=[],
                    metavar="[LABEL=]FOLDER",
                    help="add off-convention eval folder(s) to eval/sessions.json. "
                         "Accepts multiple after one flag AND is repeatable. FOLDER "
                         "is under data/<task>/ (or a path); optional LABEL sets the "
                         "bar label (use \\n for a sub-label). E.g. --extra-eval "
                         "'Blended extra\\nr1=eval_blended_round1_extra' "
                         "'Dagger extra\\nr1=eval_dagger_round1_extra'")
    args = ap.parse_args()
    global MAKE_PROC
    MAKE_PROC = not args.no_make_proc
    # nargs='+' + append gives a list-of-lists; flatten to one spec list.
    args.extra_eval = [spec for group in args.extra_eval for spec in group]

    if args.ablations and args.no_ablations:
        sys.exit("pass only one of --ablations / --no-ablations")

    def want(step):
        return step in args.only if args.only else step not in args.skip

    cfg_path = os.path.join(CONF, "analysis", "real", f"{args.config}.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(CONF, f"{args.config}.yaml")
    if not os.path.exists(cfg_path):
        sys.exit(f"config not found: {cfg_path}")
    cfg = load_task_config(cfg_path)

    task = cfg.task
    # data_dir pins the folder when it is not <data_root>/<task>; round folders keep <task>_ names.
    task_dir = str(cfg.get("data_dir", None) or os.path.join(cfg.data_root, task))
    if not os.path.isdir(task_dir):
        sys.exit(f"task dir not found: {task_dir}")
    algos = OmegaConf.to_container(cfg.algos, resolve=True)

    def method_of(key):
        """The exported method name; an ablation keeps its own."""
        return str((algos.get(key) or {}).get("method") or episode_export.method_of(key))

    req = int(cfg.num_episodes)
    ereq = int(cfg.eval_num_episodes)
    include_ablations = bool(cfg.get("include_ablations", False))
    if args.ablations:
        include_ablations = True
    if args.no_ablations:
        include_ablations = False

    print(f"{C_HEAD}== Detecting rounds under {task_dir} "
          f"(ablations {'ON' if include_ablations else 'off'}) =={C_END}")

    runs, run_folders, present, labels, colors, alpha_specs = detect_runs(
        task_dir, task, algos, req, include_ablations, cfg)

    if not present:
        sys.exit("No complete round folders detected for any algorithm.")

    # Recorded here so downstream plotting need not reopen every HDF5.
    alpha_dir = os.path.join(cfg.output_dir, "alpha")
    os.makedirs(alpha_dir, exist_ok=True)
    for s in alpha_specs:
        s["alpha_is_constant"] = _alpha_is_constant(s["full"])
        s["has_blend_fields"] = _has_blend_fields(s["full"])
    _zoom = cfg.alpha.zoom
    with open(os.path.join(alpha_dir, "sessions.json"), "w") as fh:
        json.dump(dict(
            task=task,
            zoom=dict(demo=str(_zoom.demo), seconds=_zoom.seconds),
            sessions=[dict(algo=s["algo"], round=s["n"], full=s["full"],
                           label=s["label"], color=s["color"],
                           alpha_is_constant=s["alpha_is_constant"],
                           has_blend_fields=s["has_blend_fields"])
                      for s in alpha_specs],
        ), fh, indent=2)

    # ---- eval: pretrain + per-run rounds (ablations included — their eval
    # folders follow the same eval_<run_key>_round<N> convention, e.g.
    # eval_blended_noUQ_round1). Eval rounds are independent of collection
    # rounds, so a run may have eval for rounds it has no collection folder for.
    eval_runs = {}
    # {method: {round: eval path}} for the per-episode export's rollouts.csv. The round is which
    # round's data trained the policy that was evaluated, so eval_pretrain is round 0 — the base
    # policy both methods started from, which is why it is recorded under each of them.
    rollout_spec = {}

    def _add_rollouts(key, rnd, path):
        rollout_spec.setdefault(method_of(key), {})[int(rnd)] = path

    ep = detect_eval_pretrain(task_dir, ereq)
    if ep:
        eval_runs["pretrain"] = ep
    label_of = dict(zip(present, labels))
    # cfg.eval_rounds pins eval folders per algo, same reasoning as cfg.rounds.
    _pe_cfg = cfg.get("eval_rounds", None)
    pinned_eval = OmegaConf.to_container(_pe_cfg, resolve=True) if _pe_cfg else {}
    for key in present:
        if key in pinned_eval:
            paths = []
            for name in pinned_eval[key]:
                p = name if os.path.isabs(name) else os.path.join(task_dir, name)
                score = os.path.join(p, "subtask_scores.json")
                if os.path.exists(score):
                    paths.append(score)
                else:
                    warn(f"eval_rounds.{key}: {name} has no subtask_scores.json — skipping")
            if paths:
                eval_runs[key] = paths
                print(f"  eval {label_of[key]:20s}: {len(paths)} pinned rounds")
                for i, s in enumerate(paths):
                    _add_rollouts(key, i + 1, s)
            continue
        ev = detect_eval_rounds(task_dir, key, ereq)
        if ev:
            eval_runs[key] = [s for _, s in ev]
            print(f"  eval {label_of[key]:20s}: rounds {[n for n, _ in ev]}")
            for n, s in ev:
                _add_rollouts(key, n, s)
    if ep:
        for key in present:
            _add_rollouts(key, 0, ep)

    # cfg.export_rollouts adds eval folders the eval_<algo>_round<N> convention does not reach:
    # e.g. evaluations stored beside a different task folder's data.
    _xr_cfg = cfg.get("export_rollouts", None)
    for method, rounds in (OmegaConf.to_container(_xr_cfg, resolve=True) if _xr_cfg else {}).items():
        for rnd, path in (rounds or {}).items():
            if os.path.exists(path):
                rollout_spec.setdefault(str(method), {})[int(rnd)] = path
            else:
                warn(f"export_rollouts.{method}.{rnd}: {path} not found — skipping")

    # Explicit off-convention eval folders (--extra-eval), appended to the plot.
    extra_evals = [e for e in (resolve_extra_eval(task_dir, spec, ereq)
                               for spec in args.extra_eval) if e]

    pretrain_path = os.path.join(task_dir, cfg.pretrain.folder, cfg.pretrain.dataset)
    if not os.path.exists(pretrain_path):
        warn(f"pretrain dataset not found: {pretrain_path}")

    out_dir = cfg.output_dir

    # ------------------------------------------------------------------ #
    # Eval manifest: which eval folder is which round of which method.
    #
    # Order is load-bearing: readers index rounds by position within a method, not by folder number.
    # ------------------------------------------------------------------ #
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    color_of = dict(zip(present, colors))
    eval_sessions = []
    if ep:
        eval_sessions.append(dict(algo="pretrain", round=0, scores=ep,
                                  label="Base policy", color="0.6",
                                  n_episodes=_n_eval_episodes(ep)))
    for key in present:
        entry = eval_runs.get(key)
        if not entry:
            continue
        for i, score in enumerate([entry] if isinstance(entry, str) else entry):
            m = re.search(r"_round(\d+)", os.path.basename(os.path.dirname(score)))
            eval_sessions.append(dict(algo=key, round=int(m.group(1)) if m else i + 1,
                                      scores=score, label=label_of[key],
                                      color=color_of.get(key),
                                      n_episodes=_n_eval_episodes(score)))
    _sub = cfg.get("subtask_labels", None)
    with open(os.path.join(eval_dir, "sessions.json"), "w") as fh:
        json.dump(dict(task=str(cfg.get("run_name", task)),
                       expected_eval_episodes=ereq,
                       subtask_labels=list(OmegaConf.to_container(_sub, resolve=True)) if _sub else [],
                       sessions=eval_sessions,
                       extra=[dict(label=lbl, scores=path) for lbl, path in extra_evals]),
                  fh, indent=2)
    print(f"  eval sessions → {os.path.join(eval_dir, 'sessions.json')} "
          f"({len(eval_sessions)} sessions, {len(extra_evals)} extra)")

    # ------------------------------------------------------------------ #
    # Generate the temp Hydra config for run_analysis
    # ------------------------------------------------------------------ #
    ana_name = f"_suite_{task}"
    ana_file = os.path.join(CONF, f"{ana_name}.yaml")

    analysis_cfg = OmegaConf.create({
        "defaults": ["paths_db", "_self_"],
        "hydra": {"output_subdir": None, "run": {"dir": "."}},
        "model": cfg.model, "env_name": cfg.env_name, "run_name": task,
        "control_hz": cfg.get("control_hz", 30),
        "collection_action_scale": cfg.get("collection_action_scale", 0.8),
        "runs": runs,
        "plot": {"run_names": present, "run_labels": labels, "run_colors": colors,
                 "task_name": cfg.get("task_name", task)},   # paper-figure titles
        "output_dir": out_dir,
        "plot_only": args.plot_only,
        "show_titles": not args.no_titles,
        "obs_keys": OmegaConf.to_container(cfg.obs_keys, resolve=True),
        "angle_keys": OmegaConf.to_container(cfg.get("angle_keys", []), resolve=True),
        "only_metrics": list(args.only_metrics),
        "ood": OmegaConf.to_container(cfg.ood, resolve=True),
        "trajectory": OmegaConf.to_container(cfg.trajectory, resolve=True),
        "intervention_timing": OmegaConf.to_container(cfg.intervention_timing, resolve=True),
        "paths_db": {"lookup": {cfg.model: {cfg.env_name: {"pretrained_dataset": pretrain_path}}}},
        # Per-episode export (format: episode_export.py). task/study/environment describe the data,
        # not the folder layout: the user-study cupboard run is still the cupboard task.
        "export": {
            "enabled": bool(cfg.get("export_enabled", True)),
            "dataset": str(cfg.get("export_dataset", os.path.basename(os.path.normpath(out_dir)))),
            "study": str(cfg.get("study", "expert")),
            "task": str(cfg.get("export_task", task)),
            "environment": str(cfg.get("environment", "real")),
            "methods": {key: method_of(key) for key in present},
            "rollouts": rollout_spec,
        },
    })
    written = []  # defined before try so the cleanup below cannot NameError
    try:
        OmegaConf.save(analysis_cfg, ana_file)
        written = [ana_file]

        if want("analysis"):
            run([PY, "analysis/run_analysis.py", "--config-name", ana_name])
    finally:
        for f in written:
            try:
                os.remove(f)
            except OSError:
                pass

    print(f"\n{C_HEAD}Suite complete → {out_dir}/{C_END}")


if __name__ == "__main__":
    main()
