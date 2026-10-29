"""
Joint-space jerk at rollout ↔ intervention transitions.

Jerk = ||joint_acc[t] - joint_acc[t-1]||_2, where joint_acc is the
'joint_acc' dataset written by DataWriter (sim.data.qacc at each step).

If the dataset is absent from the HDF5, compute_joint_jerk returns None and
the caller should skip this analysis entirely.

Note on DOFs: joint_acc spans all MuJoCo DOFs including object free joints.
During robot-controlled motion the robot DOFs dominate; passive object DOFs
contribute little unless an object is actively being moved.
"""

from __future__ import annotations
import numpy as np

from metrics.transition_jerk import find_transitions


def has_joint_acc(demos: list[dict]) -> bool:
    """Return True if at least one demo contains joint_acc data."""
    return any(d.get("joint_acc") is not None for d in demos)


def compute_joint_jerk(demos: list[dict]) -> dict[str, np.ndarray] | None:
    """
    Compute joint-acceleration change magnitude at rollout↔intervention transitions.

    For each pair of consecutive timesteps (t-1, t):
      jerk = ||joint_acc[t] - joint_acc[t-1]||_2

    Timesteps are classified and filtered identically to compute_transition_jerk:
      ro_to_iv / iv_to_ro / non_transition — all timesteps
      train_ro_to_iv / train_iv_to_ro / train_non_transition — train_mask=True only

    Parameters
    ----------
    demos : list of dicts with keys:
        'joint_acc'         : (T, nv) float array from sim.data.qacc, or None
        'intervention_mask' : (T,) bool array
        'train_mask'        : (T,) bool array (optional)

    Returns
    -------
    dict with six float64 arrays, or None if joint_acc is absent in all demos.
    """
    if not has_joint_acc(demos):
        return None

    jerk_ro_iv, jerk_iv_ro, jerk_base = [], [], []
    train_ro_iv, train_iv_ro, train_base = [], [], []

    for d in demos:
        ja = d.get("joint_acc")
        if ja is None:
            continue

        iv = d["intervention_mask"]
        tm = d.get("train_mask")

        ro_to_iv, iv_to_ro = find_transitions(iv)
        ro_to_iv_set = set(ro_to_iv)
        iv_to_ro_set = set(iv_to_ro)

        for t in range(1, len(iv)):
            j = float(np.linalg.norm(ja[t] - ja[t - 1]))
            in_train = tm is not None and bool(tm[t])

            if t in ro_to_iv_set:
                jerk_ro_iv.append(j)
                if in_train:
                    train_ro_iv.append(j)
            elif t in iv_to_ro_set:
                jerk_iv_ro.append(j)
                if in_train:
                    train_iv_ro.append(j)
            else:
                jerk_base.append(j)
                if in_train:
                    train_base.append(j)

    return dict(
        ro_to_iv            =np.array(jerk_ro_iv,  dtype=np.float64),
        iv_to_ro            =np.array(jerk_iv_ro,  dtype=np.float64),
        non_transition      =np.array(jerk_base,   dtype=np.float64),
        train_ro_to_iv      =np.array(train_ro_iv, dtype=np.float64),
        train_iv_to_ro      =np.array(train_iv_ro, dtype=np.float64),
        train_non_transition=np.array(train_base,  dtype=np.float64),
    )
