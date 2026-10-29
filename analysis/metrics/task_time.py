"""Task time measured over the window where the grip trigger is held.

The episode timer written into *_meta.json (`task_time_seconds`) starts when the
episode is created, but the operator may not squeeze the grip trigger — which
starts autonomous motion / enables takeover — until seconds later, and recording
continues after they let go at the end. The robot cannot move at either end (the
collector zeroes the executed action whenever the grip is released), so both
stretches are dead time that inflates the reported task time.

`corrected_task_times()` recomputes each demo's task time as the span from the
FIRST to the LAST `grip_held_mask` True:

  * with a `timestamps` dataset  → exact wall-clock, ts[last_grip] - ts[first_grip]
  * without it                   → (last_grip - first_grip) / CONTROL_HZ

The control loop runs at a fixed 30 Hz, so the step count alone gives the time.
The period is not derived from meta (task_time_seconds / num_steps) because meta's
timer also covers time outside the logged timesteps.

Only when `grip_held_mask` is missing entirely do we fall back to meta:
  * `grip_held_mask` absent — e.g. spacemouse teleop
  * the mask is never True
  * the meta row has no demo of that name in the hdf5 (e.g. restart rows)
"""
from __future__ import annotations

import os

import h5py
import numpy as np

GRIP_KEY = "grip_held_mask"
DEFAULT_CONTROL_HZ = 30.0
_CONTROL_HZ = DEFAULT_CONTROL_HZ


def set_control_hz(hz: float) -> None:
    """Override the control-loop rate used to convert timesteps → seconds."""
    global _CONTROL_HZ
    _CONTROL_HZ = float(hz) if hz else DEFAULT_CONTROL_HZ


def meta_path_for(full_path: str) -> str | None:
    """Sibling *_meta.json for a *_full.hdf5 path (or None)."""
    if not full_path.endswith("_full.hdf5"):
        return None
    p = full_path[: -len("_full.hdf5")] + "_meta.json"
    return p if os.path.exists(p) else None


def full_path_for(meta_path: str) -> str | None:
    """Sibling *_full.hdf5 for a *_meta.json path (or None)."""
    if not meta_path.endswith("_meta.json"):
        return None
    p = meta_path[: -len("_meta.json")] + "_full.hdf5"
    return p if os.path.exists(p) else None


def grip_span(grp) -> tuple[int, int] | None:
    """(first, last) timestep the grip trigger is held — the window the robot can move in.

    While the trigger is released the collector zeroes the executed action, so steps
    outside this window are dead time: at the head, how long the operator took to
    squeeze; at the tail, recording that ran on after they let go. Returns None when
    there is no grip mask (sim) or it is never pressed.
    """
    if GRIP_KEY not in grp:
        return None
    pressed = np.flatnonzero(np.asarray(grp[GRIP_KEY][:]).astype(bool).ravel())
    return (int(pressed[0]), int(pressed[-1])) if len(pressed) else None


def grip_window(grp, n: int) -> tuple[int, int]:
    """grip_span with a fallback to the whole episode [0, n-1] (sim, or no press)."""
    span = grip_span(grp)
    return span if span is not None else (0, n - 1)


def _demo_task_time(grp, meta_row: dict):
    """(seconds, corrected?) for one demo group."""
    fallback = meta_row.get("task_time_seconds")
    if GRIP_KEY not in grp:
        return fallback, False

    span = grip_span(grp)
    if span is None:
        return fallback, False

    i0, i1 = span                 # first / last timestep the grip trigger is held
    T = len(grp[GRIP_KEY])
    if i1 <= i0:
        return fallback, False

    # Exact wall-clock when timestamps were logged.
    if "timestamps" in grp:
        ts = np.asarray(grp["timestamps"][:], dtype=float).ravel()
        if len(ts) == T:
            return float(ts[i1] - ts[i0]), True

    # Otherwise use the fixed control rate — every timestep is 1/CONTROL_HZ.
    return float((i1 - i0) / _CONTROL_HZ), True


def corrected_task_times(full_path: str, meta: dict) -> tuple[dict, int, int]:
    """Return (patched_meta, n_corrected, n_total).

    `meta` is the parsed *_meta.json ({demo_name: {...}}). The returned dict is a
    shallow-copied version whose `task_time_seconds` is replaced by the
    grip-trigger-relative time wherever it can be computed.
    """
    if not os.path.exists(full_path):
        return meta, 0, len(meta)

    patched, n_corr = {}, 0
    with h5py.File(full_path, "r") as f:
        data = f["data"] if "data" in f else {}
        for demo, row in meta.items():
            row = dict(row)
            if demo in data:
                secs, ok = _demo_task_time(data[demo], row)
                if ok and secs is not None:
                    row["task_time_seconds"] = secs
                    n_corr += 1
            patched[demo] = row
    return patched, n_corr, len(meta)


def correct_meta_from_json(meta_path: str, meta: dict) -> dict:
    """Patch a meta dict given the path it was loaded from (finds the hdf5)."""
    full = full_path_for(meta_path)
    if full is None:
        return meta
    patched, n_corr, n_tot = corrected_task_times(full, meta)
    if n_corr:
        print(f"  [task_time] {os.path.basename(meta_path)}: recomputed from grip "
              f"trigger for {n_corr}/{n_tot} demos")
    return patched
