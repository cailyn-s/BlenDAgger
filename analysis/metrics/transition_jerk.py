"""
Jerk analysis at rollout ↔ intervention transitions.

"Jerk" here is the L2 norm of the action change between consecutive timesteps,
||a[t] - a[t-1]||_2, measured specifically at the moment the controller switches
vs. at all other (non-transition) timesteps as a within-episode baseline.

At a rollout→intervention transition, t-1 is the last autonomous robot step and
t is the first human correction step.  At an intervention→rollout transition,
t-1 is the last human step and t is the first autonomous step.

For BlenDAgger the 'actions' field already holds the blended output, so the
metric compares the blended handover against HG-DAgger's hard switch.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Transition detection
# ---------------------------------------------------------------------------

def find_transitions(iv_mask: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Identify timesteps where the intervention mask changes.

    Parameters
    ----------
    iv_mask : (T,) bool array — True where human was intervening

    Returns
    -------
    ro_to_iv : list of int — index t where intervention starts (iv_mask flips F→T)
    iv_to_ro : list of int — index t where rollout resumes  (iv_mask flips T→F)
    """
    ro_to_iv, iv_to_ro = [], []
    for t in range(1, len(iv_mask)):
        if not iv_mask[t - 1] and iv_mask[t]:
            ro_to_iv.append(t)
        elif iv_mask[t - 1] and not iv_mask[t]:
            iv_to_ro.append(t)
    return ro_to_iv, iv_to_ro


# ---------------------------------------------------------------------------
# Jerk computation
# ---------------------------------------------------------------------------

def compute_transition_jerk(demos: list[dict]) -> dict[str, np.ndarray]:
    """
    Compute action-change magnitude at rollout↔intervention transitions.

    For each pair of consecutive timesteps (t-1, t):
      jerk = ||actions[t] - actions[t-1]||_2

    Timesteps are classified into three groups:
      ro_to_iv       — t is the first step of a human intervention
      iv_to_ro       — t is the first step of autonomous rollout after intervention
      non_transition — all other consecutive pairs (within-episode baseline)

    Each group is computed twice: once over all timesteps, and once filtered to
    timesteps where train_mask[t] is True (the IWR training set).

    Parameters
    ----------
    demos : list of dicts with keys:
        'actions'           : (T, A) float array of executed actions
        'intervention_mask' : (T,) bool array
        'train_mask'        : (T,) bool array (optional; filtered stats omitted if absent)

    Returns
    -------
    dict with float64 arrays:
        ro_to_iv             : jerk at each ro→iv switch (all timesteps)
        iv_to_ro             : jerk at each iv→ro switch (all timesteps)
        non_transition       : jerk at all non-transition pairs (all timesteps)
        train_ro_to_iv       : same, filtered to train_mask[t]=True
        train_iv_to_ro       : same, filtered to train_mask[t]=True
        train_non_transition : same, filtered to train_mask[t]=True
    """
    jerk_ro_iv, jerk_iv_ro, jerk_base = [], [], []
    train_ro_iv, train_iv_ro, train_base = [], [], []

    for d in demos:
        iv = d["intervention_mask"]
        a  = d["actions"]              # (T, A)

        if "actions" not in d or a is None:
            raise KeyError(
                "Demo dict missing 'actions'. "
                "Ensure load_collected_states is called with load_actions=True."
            )

        tm = d.get("train_mask")       # (T,) bool or None

        ro_to_iv, iv_to_ro = find_transitions(iv)
        ro_to_iv_set = set(ro_to_iv)
        iv_to_ro_set = set(iv_to_ro)

        for t in range(1, len(iv)):
            j = float(np.linalg.norm(a[t] - a[t - 1]))
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

def compute_joint_accel(demos: list[dict], hz: float = 30.0) -> dict[str, np.ndarray]:
    """Joint-space discontinuity at rollout<->intervention handoffs, in rad/s^2.

    The command-space version norms the recorded 7-D action, which mixes linear
    velocity fraction, angular velocity fraction and an absolute gripper position.
    This is the same quantity in one physical unit: the change in realised joint
    velocity, ||qdot_t - qdot_{t-1}|| * hz, with qdot_t = wrap(q_t - q_{t-1}) * hz.
    Angles are recovered from the (sin, cos) state and differences are wrapped, so
    a joint crossing +/-pi is not counted as a huge jump. It measures the REALISED
    motion rather than the command, which for a tracking servo is nearly the same.
    """
    out = {k: [] for k in ("ro_to_iv", "iv_to_ro", "non_transition")}
    for d in demos:
        iv = d["intervention_mask"]
        st = d["states"]
        J = st.shape[1] // 2
        th = np.arctan2(st[:, :J], st[:, J:])
        if len(th) < 3:
            continue
        dq = np.arctan2(np.sin(np.diff(th, axis=0)), np.cos(np.diff(th, axis=0))) * hz
        acc = np.linalg.norm(np.diff(dq, axis=0), axis=1) * hz   # index i -> timestep i+2
        ro_to_iv, iv_to_ro = find_transitions(iv)
        ro_set, iv_set = set(ro_to_iv), set(iv_to_ro)
        for i, val in enumerate(acc):
            t = i + 2
            key = ("ro_to_iv" if t in ro_set else
                   "iv_to_ro" if t in iv_set else "non_transition")
            out[key].append(float(val))
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}

def compute_command_delta(demos: list[dict], action_scale: float = 0.8
                          ) -> dict[str, np.ndarray]:
    """Commanded-motion discontinuity, split into linear and angular parts.

    The 7-D recorded action is NOT normalised: the linear and angular blocks have
    different scales, so norming all seven together is dominated by the rotation
    channel, and the seventh entry is an absolute gripper position rather than a
    velocity. Here the two velocity blocks are converted to physical
    units and reported separately, so nothing incommensurable is added:

        lin_*  ||dv||  in m/s     (1.0 command = 0.20 * action_scale m/s)
        ang_*  ||dw||  in rad/s   (1.0 command = pi/8 * action_scale rad/s)

    The gripper channel is excluded from both.
    """
    LIN, ANG = 0.20 * action_scale, (np.pi / 8.0) * action_scale
    out = {f"{p}_{k}": [] for p in ("lin", "ang")
           for k in ("ro_to_iv", "iv_to_ro", "non_transition")}
    for d in demos:
        iv = d["intervention_mask"]
        a = d["actions"]
        if a is None or len(a) < 2:
            continue
        v, w = a[:, :3] * LIN, a[:, 3:6] * ANG
        dv = np.linalg.norm(np.diff(v, axis=0), axis=1)
        dw = np.linalg.norm(np.diff(w, axis=0), axis=1)
        ro_to_iv, iv_to_ro = find_transitions(iv)
        ro_set, iv_set = set(ro_to_iv), set(iv_to_ro)
        for i in range(len(dv)):
            t = i + 1
            k = ("ro_to_iv" if t in ro_set else
                 "iv_to_ro" if t in iv_set else "non_transition")
            out[f"lin_{k}"].append(float(dv[i]))
            out[f"ang_{k}"].append(float(dw[i]))
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def compute_during_intervention_delta(demos: list[dict]) -> list[float]:
    """Mean ||a[t] - a[t-1]|| with t-1 and t BOTH inside the same intervention segment.

    One value per demo, in the order given, and NaN for a demo with no intervention segment of
    length two or more. This is the "smoother corrections" reading of action change: the delta at
    a handover boundary is partly mechanical, because the arbitration ramps alpha across it, so
    the transition steps themselves are excluded here and so is everything outside an
    intervention.

    `transition_jerk`'s `train_non_transition` is adjacent but not the same: it filters on
    `train_mask`, the IWR training set, which is a different set of timesteps from the
    intervention mask.
    """
    out = []
    for d in demos:
        iv = np.asarray(d["intervention_mask"], dtype=bool)
        a = d.get("actions")
        if a is None or len(a) < 2:
            out.append(float("nan"))
            continue
        inside = iv[1:] & iv[:-1]              # t-1 and t both intervening = same segment
        if not inside.any():
            out.append(float("nan"))
            continue
        deltas = np.linalg.norm(np.diff(np.asarray(a, dtype=np.float64), axis=0), axis=1)
        out.append(float(deltas[inside].mean()))
    return out
