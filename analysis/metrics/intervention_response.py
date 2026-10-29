"""
How quickly, and how completely, the robot follows a human correction.

Autonomous success rate says nothing about what an intervention *feels* like. These three
families measure that directly, all over the human's linear (translation) command, which is
the channel the operator judges responsiveness by:

  response latency — from the moment the operator starts commanding motion inside an
      intervention, how many timesteps until the robot goes where they are pointing. Measured
      twice: in COMMAND space (the arbitrated action leaving the blender vs. the human action
      entering it) and in ACHIEVED space (the arm's real end-effector velocity vs. the same
      human action). The command-space number isolates the arbitration; the achieved-space
      number adds the hardware's velocity ramp and is what the operator actually sees.

  stalls — steps inside an intervention where the operator is asking for motion and the arm is
      essentially not moving. Reported as a step fraction, as discrete events, and as event
      durations.

  magnitude — the human input's own size, summarised by its MAX per episode as well as the mean
      the collector already logs. A method that is hard to steer is one the operator has to push
      hard against, and the peak of that push is not visible in the mean. The collector's own
      magnitude clips every DoF group at 1.0, so its maximum saturates at sqrt(3); the unclipped
      translation norm and the fraction of steps spent past full speed are reported alongside it.

Latency answers "how long before it starts obeying"; the tracking fraction reported with it
answers "and did it keep obeying" — the same alignment test applied to every commanding step
rather than only the first, which is what catches a method that fights throughout instead of
just at the handover.

Onset not-reached
-----------------
The fraction of interventions in which the arbitrated command never comes within the direction
cone (and, optionally, the speed band) of the human command within a deadline after the operator
starts commanding; see compute_response and its `horizon` / `min_commanding` arguments.

Frames and units
----------------
`human_actions[:3]` and `actions[:3]` are normalised world-frame linear velocity commands: the
driver multiplies them by `max_lin_vel = 0.20 * action_scale` m/s and norm-limits the result at
that cap, so a command of norm >= 1 asks for full speed and direction is preserved exactly.
`obs/arm_pos` is the end-effector position in the same world frame, so a finite difference of it
is directly comparable to both. Achieved velocity uses a FORWARD difference over
`vel_window` steps because consecutive `arm_pos` reads on this hardware can repeat the previous
value (the SDK poll is slower than the control loop); a single-step difference would read those
as stalls. The window costs the latency numbers up to `vel_window` steps of resolution.
"""

from __future__ import annotations

import numpy as np

from metrics import intervention_timing as itiming

# Hardware speed caps before action_scale (xArm driver in the companion repository).
MAX_LINEAR_VEL = 0.20
MAX_ANGULAR_VEL = np.pi / 8

# Which DoFs a metric is measured over. "both" scores a step only when EVERY channel the
# operator is driving at that step is matched; a channel they are not driving is vacuous.
CHANNELS = ("linear", "angular", "both")
CHANNEL = "linear"

# How the platform applies a rotation command, which decides how to recover the achieved
# angular velocity from a pose log. Measured per platform, not assumed:
#   "body"  right-multiply, R <- R * dR   (xArm driver, in the SDK frame)
#   "world" left-multiply,  R <- dR * R   (robosuite OSC_POSE, in the base frame)
# The wrong choice flips the recovered axis.
ANGULAR_FRAME = "body"
# Yaw between the pose log's frame and the frame the command is given in (xArm: the SDK frame is
# 90 deg from the world frame the observation uses; sim: none).
ANGULAR_FRAME_YAW = np.pi / 2

# A command smaller than this fraction of full speed is not the operator steering; it is
# deadzone residue or a hold-still frame, and neither latency nor a stall is defined there.
CMD_FLOOR = 0.10
# "Mostly in the direction they are commanding": cosine between the two unit vectors.
COS_THRESH = 0.90
# Moving at all: clear an absolute per-channel noise floor AND a fraction of what was asked.
MOVE_ABS_FLOOR = 0.005          # m/s
MOVE_ABS_FLOOR_ANG = 0.025      # rad/s
MOVE_RATIO = 0.20
# A STALL is defined against the floors alone, not against what was asked: the arm counts as
# stalled when NEITHER channel clears its own floor by more than STALL_FLOOR_MARGIN - 1. Keeping
# the command out of it means the metric answers "is the robot moving at all while I am asking
# it to" rather than "is it moving as fast as I asked", which is what the match metrics measure.
STALL_FLOOR_MARGIN = 1.10
# A stall shorter than this is the velocity ramp between direction changes, not a stall.
MIN_STALL_STEPS = 3
VEL_WINDOW = 3


def intervention_magnitude(ha6: np.ndarray) -> np.ndarray:
    """Per-step human input size, the collector's own definition, vectorised over (T, 6).

    Each DoF group is clipped at 1.0 before the groups are combined, so the value is bounded
    and device-independent. Must stay identical to `intervention_magnitude` in the xArm data
    writer in the companion repository for the recomputed mean to match the logged one.
    """
    ha6 = np.asarray(ha6, dtype=np.float64)
    return np.linalg.norm(np.stack([
        np.minimum(np.linalg.norm(ha6[:, 0:2], axis=1), 1.0),
        np.minimum(np.abs(ha6[:, 2]), 1.0),
        np.minimum(np.linalg.norm(ha6[:, 3:6], axis=1), 1.0),
    ], axis=1), axis=1)


def achieved_velocity(arm_pos: np.ndarray, hz: float, window: int | None = None) -> np.ndarray:
    """(T, 3) end-effector velocity in m/s; the last `window` steps are NaN.

    `window` is read at call time, not bound as a default, so overriding the module constant in
    a sensitivity sweep actually takes effect.
    """
    window = VEL_WINDOW if window is None else int(window)
    arm_pos = np.asarray(arm_pos, dtype=np.float64)
    vel = np.full(arm_pos.shape, np.nan)
    if len(arm_pos) > window:
        vel[:-window] = (arm_pos[window:] - arm_pos[:-window]) / (window / float(hz))
    return vel


def _unit(vecs: np.ndarray, norms: np.ndarray) -> np.ndarray:
    return vecs / np.maximum(norms, 1e-12)[:, None]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_excl) of each contiguous True run."""
    return itiming.extract_boundaries(np.asarray(mask, dtype=bool))


def achieved_angular_velocity(arm_quat: np.ndarray, hz: float,
                              window: int | None = None) -> np.ndarray:
    """(T, 3) angular velocity in rad/s, in the frame and composition the command uses.

    The pose log is a world-frame orientation; the command is a rotation delta the platform
    applies in its own frame and on its own side. Both are configurable because they differ:
    the xArm driver does `R_sdk <- R_sdk * dR` (body, right, in a frame yawed 90 deg from the
    observation), robosuite's OSC_POSE does `R <- dR * R` (world, left, no offset). The wrong
    choice flips the recovered axis, so both are set per platform from measurement.
    """
    from scipy.spatial.transform import Rotation

    window = VEL_WINDOW if window is None else int(window)
    quat = np.asarray(arm_quat, dtype=np.float64)
    out = np.full((len(quat), 3), np.nan)
    if len(quat) <= window:
        return out
    norms = np.linalg.norm(quat, axis=1)
    if not np.all(np.isfinite(norms) & (norms > 1e-6)):
        return out                                  # unusable quaternions; contribute nothing
    rot = Rotation.from_quat(quat / norms[:, None])
    if ANGULAR_FRAME_YAW:
        rot = Rotation.from_euler("z", ANGULAR_FRAME_YAW).inv() * rot
    delta = (rot[:-window].inv() * rot[window:]) if ANGULAR_FRAME == "body" \
        else (rot[window:] * rot[:-window].inv())
    out[:-window] = delta.as_rotvec() / (window / float(hz))
    return out


def _channel_signals(human: np.ndarray, exec_a: np.ndarray, achieved: np.ndarray,
                     cap: float, abs_floor: float) -> dict:
    """Per-step commanding / direction / match arrays for ONE channel of the action.

    `achieved` arrives in physical units and `cap` converts it into the same normalised units
    the command uses, so one match test serves translation and rotation alike.
    """
    u_norm = np.linalg.norm(human, axis=1)
    u_hat = _unit(human, u_norm)
    # The driver norm-limits at the cap, so anything past 1.0 asks for exactly full speed.
    asked = np.minimum(u_norm, 1.0)
    ach = achieved / cap
    ok = np.isfinite(ach).all(axis=1)
    ach = np.nan_to_num(ach)
    ach_speed = np.linalg.norm(ach, axis=1)
    e_norm = np.linalg.norm(exec_a, axis=1)
    return dict(
        u_norm=u_norm,
        asked=asked,
        active=u_norm >= CMD_FLOOR,
        ok=ok,
        ach_speed=ach_speed,
        cap=float(cap),
        abs_floor=float(abs_floor),
        moving=ok & (ach_speed >= np.maximum(abs_floor / cap, MOVE_RATIO * asked)),
        cos_cmd=np.where(e_norm > 1e-9,
                         (exec_a * u_hat).sum(axis=1) / np.maximum(e_norm, 1e-12), -1.0),
        cos_ach=np.where(ok & (ach_speed > 1e-9),
                         (ach * u_hat).sum(axis=1) / np.maximum(ach_speed, 1e-12), -1.0),
        # Direction AND speed: the delivered motion must land within MATCH_TOL of the speed
        # asked for, along the direction asked for. A policy pushing back along the operator's
        # own axis leaves the direction intact and only changes the speed, so a direction-only
        # test scores both resistance and overshoot as obedience and this does not. Both sides
        # pass through the driver's norm limit, so the comparison is what the arm can be told.
        match_cmd=_match_mask(limit_norm(exec_a), u_hat, asked, MATCH_TOL),
        match_ach=ok & _match_mask(limit_norm(ach), u_hat, asked, MATCH_TOL),
    )


def _demo_signals(demo: dict, hz: float, action_scale: float,
                  channel: str | None = None) -> dict | None:
    """Per-step booleans and cosines the metric families share, for the chosen channel(s).

    `channel` is "linear" (translation), "angular" (rotation) or "both". Under "both" a step
    counts as commanding when EITHER channel is being driven, and as matched only when every
    channel being driven is matched — a channel the operator is not touching cannot fail.
    """
    channel = CHANNEL if channel is None else channel
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    need = ("human_actions", "actions") + (("arm_pos",) if channel != "angular" else ())
    if any(demo.get(k) is None for k in need):
        return None
    if channel != "linear" and demo.get("arm_quat") is None:
        return None
    human = np.asarray(demo["human_actions"], dtype=np.float64)[:, :6]
    exec_a = np.asarray(demo["actions"], dtype=np.float64)[:, :6]
    iv = np.asarray(demo["intervention_mask"], dtype=bool)
    lengths = [len(human), len(exec_a), len(iv)]
    if channel != "angular":
        arm_pos = np.asarray(demo["arm_pos"], dtype=np.float64)
        lengths.append(len(arm_pos))
    if channel != "linear":
        arm_quat = np.asarray(demo["arm_quat"], dtype=np.float64)
        lengths.append(len(arm_quat))
    T = min(lengths)
    human, exec_a, iv = human[:T], exec_a[:T], iv[:T]

    parts = {}
    if channel != "angular":
        parts["linear"] = _channel_signals(
            human[:, :3], exec_a[:, :3], achieved_velocity(arm_pos[:T], hz),
            MAX_LINEAR_VEL * float(action_scale), MOVE_ABS_FLOOR)
    if channel != "linear":
        parts["angular"] = _channel_signals(
            human[:, 3:6], exec_a[:, 3:6], achieved_angular_velocity(arm_quat[:T], hz),
            MAX_ANGULAR_VEL * float(action_scale), MOVE_ABS_FLOOR_ANG)

    used = list(parts)
    active = np.any([parts[c]["active"] for c in used], axis=0)
    # A channel the operator is not driving is vacuously satisfied, so "both" does not punish
    # a pure translation for the rotation the operator never asked for.
    def _all_driven(key):
        return np.all([~parts[c]["active"] | parts[c][key] for c in used], axis=0)

    primary = parts[used[0]]
    return dict(
        pos_magnitude=primary["u_norm"],
        # Asking for more than the arm can deliver: the driver norm-limits at 1.0, so every
        # step past it is effort the operator spent that the robot could not act on.
        saturated=primary["u_norm"] >= 1.0,
        iv=iv,
        commanding=iv & active,
        moving=_all_driven("moving"),
        # A STALL is the arm sitting still, so it needs EVERY channel to be still — rotating
        # while translation is blocked is not a stall. Purely absolute: each channel must fail
        # to clear its own noise floor by the margin. That is a different question from
        # `moving` above (which is relative to the command and is what alignment needs), so the
        # two cannot share one key.
        moving_any=np.any([(parts[c]["ach_speed"] * parts[c]["cap"]
                            >= STALL_FLOOR_MARGIN * parts[c]["abs_floor"]) & parts[c]["ok"]
                           for c in used], axis=0),
        # The two halves of "moving", kept apart so each can be reported on its own:
        #   rel — is the arm delivering a fifth of the speed that was ASKED for
        #   abs — is the arm clearing its own physical noise floor at all
        # Both aggregate as "any channel", so a stall needs every channel to fail.
        moving_rel_any=np.any([parts[c]["active"] & parts[c]["ok"]
                               & (parts[c]["ach_speed"] >= MOVE_RATIO * parts[c]["asked"])
                               for c in used], axis=0),
        moving_abs_any=np.any([parts[c]["ok"] & (parts[c]["ach_speed"] * parts[c]["cap"]
                                                 >= parts[c]["abs_floor"])
                               for c in used], axis=0),
        vel_ok=np.all([parts[c]["ok"] for c in used], axis=0),
        cos_cmd=primary["cos_cmd"],
        cos_ach=primary["cos_ach"],
        magnitude=intervention_magnitude(human),
        match_cmd=_all_driven("match_cmd"),
        match_ach=np.all([parts[c]["ok"] for c in used], axis=0) & _all_driven("match_ach"),
        channels=parts,
        grip_window=demo.get("grip_window", (0, T - 1)),
        n_steps=T,
    )


def limit_norm(vecs: np.ndarray) -> np.ndarray:
    """The driver's own norm limit: scale back to unit norm, preserving direction.

    xarm_driver.step multiplies the action by max_lin_vel and norm-limits the result at that
    cap, so an arbitrated action of norm 3 reaches the arm as the same direction at norm 1. A
    two-sided speed test that skipped it would fail those steps for an overshoot the robot never
    executes.
    """
    norm = np.linalg.norm(vecs, axis=1)
    scale = np.where(norm > 1.0, 1.0 / np.maximum(norm, 1e-12), 1.0)
    return vecs * scale[:, None]


def _match_mask(vecs: np.ndarray, u_hat: np.ndarray, asked: np.ndarray,
                tol: float | None) -> np.ndarray:
    """(T,) bool: each row of `vecs` runs along its `u_hat`, optionally at the asked speed.

    The two halves test different failures: the cosine catches motion pointed somewhere else,
    the band catches motion that is too slow (a policy resisting along the operator's own axis)
    AND motion that is too fast (a policy dragging them past their target). `tol=None` drops the
    speed half, leaving a direction-only test.
    """
    along = (vecs * u_hat).sum(axis=1)
    norm = np.linalg.norm(vecs, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (norm > 1e-9) & (along / np.maximum(norm, 1e-12) >= COS_THRESH)
        if tol is not None:
            ok &= (along >= (1.0 - tol) * asked) & (along <= (1.0 + tol) * asked)
    return ok


def compute_response(
    demos: list[dict],
    *,
    hz: float = 30.0,
    gap_threshold: int = 30,
    action_scale: float = 0.8,
    channel: str | None = None,
    horizon: int | None = None,
    min_commanding: int | None = None,
) -> dict[str, np.ndarray]:
    """Latency, stall and magnitude arrays pooled over `demos`.

    Parameters
    ----------
    demos         : dicts from run_analysis.load_collected_states — needs `human_actions`,
                    `actions`, `arm_pos`, `intervention_mask` and optionally `grip_window`.
                    A demo missing the human input or the arm pose contributes nothing.
    gap_threshold : merge window in timesteps, the same one intervention_timing uses, so an
                    "intervention" here is the same object the timing figures count.
    action_scale  : the collection-time action scale; sets the m/s a command of norm 1 asks for.
    channel       : "linear", "angular" or "both" (default: the module's CHANNEL).
    min_commanding: an intervention is scored only when the operator issues at least this many
                    commanding steps in it — see MIN_COMMANDING_STEPS for why.
    horizon       : how many timesteps after the operator starts commanding the criterion is
                    still allowed to be met. None (the default) means "until this intervention
                    ends" — the operator has stopped correcting by then, so there is nothing
                    left to obey; `interv_window` records how long each of those windows was.

    Returns
    -------
    Per-intervention arrays (`latency_*`), per-stall-event arrays (`stall_event_*`) and
    per-episode arrays (everything else). Latencies are in timesteps; a `*_censored` entry
    holds the interventions that never reached the criterion, recorded as the number of steps
    they ran for, which is a lower bound on their latency and must not be pooled with the rest.
    """
    out = {k: [] for k in (
        "latency_cmd", "latency_ach", "latency_cmd_censored", "latency_ach_censored",
        "latency_cmd_match", "latency_ach_match", "latency_cmd_match_censored",
        "latency_ach_match_censored", "onset_strength",
        "onset_latency_ach_match", "onset_latency_cmd_match",
        "stall_event_steps", "ep_stall_frac", "ep_stall_frac_rel", "ep_stall_frac_rel10",
        "ep_stall_frac_abs",
        "ep_stall_events_per_min",
        "ep_stall_events_per_intervention", "ep_max_magnitude", "ep_p95_magnitude",
        "ep_mean_magnitude", "ep_max_magnitude_moving", "ep_unresolved_frac_cmd",
        "ep_unresolved_frac_ach", "ep_unresolved_frac_cmd_match",
        "ep_unresolved_frac_ach_match", "ep_n_interventions_scored",
        "ep_tracking_frac_cmd", "ep_tracking_frac_ach", "ep_max_pos_magnitude", "interv_window",
        "interv_n_commanding", "ep_n_interventions_dropped",
        "ep_mean_pos_magnitude", "ep_saturated_frac",
    )}
    min_commanding = MIN_COMMANDING_STEPS if min_commanding is None else int(min_commanding)
    for demo in demos:
        sig = _demo_signals(demo, hz, action_scale, channel)
        if sig is None:
            continue
        t0, t1 = sig["grip_window"]
        groups = _bounded_groups(sig["iv"], t0, t1, gap_threshold)

        n_scored, n_dropped = 0, 0
        unresolved = {"cmd": 0, "ach": 0, "cmd_match": 0, "ach_match": 0}
        for gs, ge in groups:
            span = slice(gs, ge)
            cmd_idx = np.flatnonzero(sig["commanding"][span])
            if not len(cmd_idx):
                continue                       # the operator drove no scored channel here
            start = cmd_idx[0]
            # The window the criterion has to be met in: to the end of the intervention by
            # default, or `horizon` steps from the first commanding step, whichever is shorter.
            stop = (ge - gs) if horizon is None else min(ge - gs, start + int(horizon))
            # Counted over the whole intervention so the trim and deadline stay independent.
            n_commanding = int((cmd_idx >= start).sum())
            if n_commanding < min_commanding:
                n_dropped += 1                 # too few chances to obey; see MIN_COMMANDING_STEPS
                continue
            n_scored += 1
            out["interv_n_commanding"].append(float(n_commanding))
            out["interv_window"].append(float(stop - start))
            aligned_cmd = sig["commanding"][span] & (sig["cos_cmd"][span] >= COS_THRESH)
            aligned_ach = (sig["commanding"][span] & sig["moving"][span]
                           & (sig["cos_ach"][span] >= COS_THRESH))
            out["onset_strength"].append(float(np.linalg.norm(
                np.asarray(demo["human_actions"], dtype=np.float64)[gs + start, :3])))
            resolved = {}
            for key, aligned in (("cmd", aligned_cmd), ("ach", aligned_ach),
                                 ("cmd_match", sig["commanding"][span] & sig["match_cmd"][span]),
                                 ("ach_match", sig["commanding"][span] & sig["match_ach"][span])):
                hit = np.flatnonzero(aligned[start:stop])
                if len(hit):
                    out[f"latency_{key}"].append(float(hit[0]))
                    resolved[key] = float(hit[0])
                else:
                    out[f"latency_{key}_censored"].append(float(stop - start))
                    resolved[key] = np.nan
                    unresolved[key] += 1
            # One row per onset in onset order, NaN where it never resolved, so a figure can
            # split onsets by how hard the operator pushed to open them.
            out["onset_latency_ach_match"].append(resolved["ach_match"])
            out["onset_latency_cmd_match"].append(resolved["cmd_match"])

        # Stalls are counted over the whole episode's commanding steps, not per group: a stall
        # that spans a brief mask dropout is one stall, and grouping would split it.
        window = np.zeros(sig["n_steps"], dtype=bool)
        window[t0:t1 + 1] = True
        commanding = sig["commanding"] & window & sig["vel_ok"]
        stalled = commanding & ~sig["moving_any"]
        events = [(s, e) for s, e in _runs(stalled) if e - s >= MIN_STALL_STEPS]
        n_cmd = int(commanding.sum())
        out["stall_event_steps"].extend(float(e - s) for s, e in events)
        # The fraction is every stalled step, with no run-length filter: the run filter is only
        # about what counts as a discrete EVENT, and applying it here would silently discard
        # short stalls from the headline number.
        out["ep_stall_frac"].append(int(stalled.sum()) / n_cmd if n_cmd else np.nan)
        out["ep_stall_frac_rel"].append(
            int((commanding & ~sig["moving_rel_any"]).sum()) / n_cmd if n_cmd else np.nan)
        moving_rel10 = np.any([sig["channels"][c]["active"] & sig["channels"][c]["ok"]
                               & (sig["channels"][c]["ach_speed"] >= 0.10 * sig["channels"][c]["asked"])
                               for c in sig["channels"]], axis=0)
        out["ep_stall_frac_rel10"].append(
            int((commanding & ~moving_rel10).sum()) / n_cmd if n_cmd else np.nan)
        out["ep_stall_frac_abs"].append(
            int((commanding & ~sig["moving_abs_any"]).sum()) / n_cmd if n_cmd else np.nan)
        out["ep_stall_events_per_min"].append(
            len(events) / (n_cmd / hz / 60.0) if n_cmd else np.nan)
        out["ep_stall_events_per_intervention"].append(
            len(events) / n_scored if n_scored else np.nan)
        for key in ("cmd", "ach", "cmd_match", "ach_match"):
            out[f"ep_unresolved_frac_{key}"].append(
                unresolved[key] / n_scored if n_scored else np.nan)
        out["ep_n_interventions_scored"].append(float(n_scored))
        out["ep_n_interventions_dropped"].append(float(n_dropped))

        mag = sig["magnitude"][sig["iv"] & window]
        out["ep_max_magnitude"].append(float(mag.max()) if mag.size else np.nan)
        out["ep_p95_magnitude"].append(float(np.percentile(mag, 95)) if mag.size else np.nan)
        out["ep_mean_magnitude"].append(float(mag.mean()) if mag.size else np.nan)
        mag_mov = sig["magnitude"][sig["commanding"] & window]
        out["ep_max_magnitude_moving"].append(float(mag_mov.max()) if mag_mov.size else np.nan)
        pos_mag = sig["pos_magnitude"][commanding]
        out["ep_max_pos_magnitude"].append(float(pos_mag.max()) if pos_mag.size else np.nan)
        out["ep_mean_pos_magnitude"].append(float(pos_mag.mean()) if pos_mag.size else np.nan)
        out["ep_saturated_frac"].append(
            float(sig["saturated"][commanding].mean()) if n_cmd else np.nan)

        # Steady-state counterpart of the latency: of every step the operator was steering,
        # how many did the robot spend going where they pointed.
        tracked_cmd = commanding & (sig["cos_cmd"] >= COS_THRESH)
        tracked_ach = commanding & sig["moving"] & (sig["cos_ach"] >= COS_THRESH)
        out["ep_tracking_frac_cmd"].append(tracked_cmd.sum() / n_cmd if n_cmd else np.nan)
        out["ep_tracking_frac_ach"].append(tracked_ach.sum() / n_cmd if n_cmd else np.nan)

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _bounded_groups(iv: np.ndarray, t0: int, t1: int, gap_threshold: int):
    """Merged intervention groups clipped to the grip window, as intervention_timing does."""
    bounds = [(max(s, t0), min(e, t1 + 1)) for s, e in _runs(iv)]
    bounds = [(s, e) for s, e in bounds if e > s]
    return itiming.merge_groups(bounds, gap_threshold)


def episode_row(demo: dict, *, hz: float = 30.0, gap_threshold: int = 30,
                action_scale: float = 0.8) -> dict:
    """The per-episode export columns for ONE demo, or {} when it carries no human input.

    Latency is summarised per episode by the MEAN over that episode's interventions, matching
    how the timing columns already reduce an episode; the median and max are properties of the
    pooled distribution and are reported by `summarise` instead.
    """
    res = compute_response([demo], hz=hz, gap_threshold=gap_threshold, action_scale=action_scale)
    if not len(res["ep_n_interventions_scored"]):
        return {}

    def one(key):
        arr = res[key]
        return float(arr[0]) if len(arr) and np.isfinite(arr[0]) else None

    def mean_steps(key):
        arr = res[key]
        return float(arr.mean()) / hz if arr.size else None

    return {
        "response_latency_cmd_s": mean_steps("latency_cmd"),
        "response_latency_ach_s": mean_steps("latency_ach"),
        "response_latency_cmd_match_s": mean_steps("latency_cmd_match"),
        "response_latency_ach_match_s": mean_steps("latency_ach_match"),
        "response_unresolved_frac_cmd": one("ep_unresolved_frac_cmd"),
        "response_unresolved_frac_ach": one("ep_unresolved_frac_ach"),
        "intervention_stall_frac": one("ep_stall_frac"),
        "intervention_stall_events_per_min": one("ep_stall_events_per_min"),
        "intervention_stall_events_per_intervention": one("ep_stall_events_per_intervention"),
        "max_intervention_magnitude": one("ep_max_magnitude"),
        "p95_intervention_magnitude": one("ep_p95_magnitude"),
        "max_intervention_magnitude_moving": one("ep_max_magnitude_moving"),
        "max_intervention_pos_magnitude": one("ep_max_pos_magnitude"),
        "mean_intervention_pos_magnitude": one("ep_mean_pos_magnitude"),
        "intervention_saturated_frac": one("ep_saturated_frac"),
        "intervention_tracking_frac_cmd": one("ep_tracking_frac_cmd"),
        "intervention_tracking_frac_ach": one("ep_tracking_frac_ach"),
    }


def _censored_frac(res: dict, key: str) -> float:
    """Share of interventions that never reached the criterion at all."""
    n_ok, n_no = len(res[key]), len(res[f"{key}_censored"])
    return (n_no / (n_ok + n_no)) if (n_ok + n_no) else np.nan


def summarise(res: dict[str, np.ndarray], hz: float = 30.0) -> dict:
    """Mean / median / max of each pooled family, in the units the metric is read in."""
    def stats(arr, scale=1.0):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if not arr.size:
            return dict(n=0, mean=np.nan, median=np.nan, max=np.nan)
        return dict(n=int(arr.size), mean=float(arr.mean()) * scale,
                    median=float(np.median(arr)) * scale, max=float(arr.max()) * scale)

    n_res = len(res["latency_cmd"])
    n_cens = len(res["latency_cmd_censored"])
    return dict(
        latency_cmd_steps=stats(res["latency_cmd"]),
        latency_cmd_s=stats(res["latency_cmd"], 1.0 / hz),
        latency_ach_steps=stats(res["latency_ach"]),
        latency_ach_s=stats(res["latency_ach"], 1.0 / hz),
        latency_cmd_match_s=stats(res["latency_cmd_match"], 1.0 / hz),
        latency_ach_match_s=stats(res["latency_ach_match"], 1.0 / hz),
        unresolved_cmd_match=_censored_frac(res, "latency_cmd_match"),
        unresolved_ach_match=_censored_frac(res, "latency_ach_match"),
        # Interventions the arbitrated command / the arm never brought into line at all. They
        # are excluded from the latency above, so a method can look fast here by failing more.
        unresolved_cmd=(n_cens / (n_res + n_cens)) if (n_res + n_cens) else np.nan,
        unresolved_ach=(len(res["latency_ach_censored"])
                        / max(len(res["latency_ach"]) + len(res["latency_ach_censored"]), 1)),
        stall_frac=stats(res["ep_stall_frac"]),
        stall_events_per_min=stats(res["ep_stall_events_per_min"]),
        stall_events_per_intervention=stats(res["ep_stall_events_per_intervention"]),
        stall_event_s=stats(res["stall_event_steps"], 1.0 / hz),
        max_magnitude=stats(res["ep_max_magnitude"]),
        mean_magnitude=stats(res["ep_mean_magnitude"]),
        max_pos_magnitude=stats(res["ep_max_pos_magnitude"]),
        saturated_frac=stats(res["ep_saturated_frac"]),
        tracking_frac_cmd=stats(res["ep_tracking_frac_cmd"]),
        tracking_frac_ach=stats(res["ep_tracking_frac_ach"]),
    )


# ===========================================================================
# Per-step match: the delay from EVERY commanding step, not just the onset
# ===========================================================================
# The clock restarts at every commanding step; a match needs direction AND speed (MATCH_TOL).

# Delivered speed along the commanded direction must land inside +/- MATCH_TOL of the commanded
# speed. Two-sided so that overshooting the commanded speed also fails.
MATCH_TOL = 0.10
# The reference command is stale once the operator has turned this far from it; the step is then
# censored as "they changed their mind" rather than counted as a failure to obey.
HOLD_COS = 0.80
# Give up after a second — past that the operator has re-planned, whatever the cosine says.
MATCH_HORIZON = 30
# Push-strength band for gentle corrections, used to stratify the per-step match.
GENTLE_BAND = (0.10, 0.30)

# Shorter interventions only see the arbitration's settling transient, so they are dropped.
MIN_COMMANDING_STEPS = 7


def compute_step_match(
    demos: list[dict],
    *,
    hz: float = 30.0,
    action_scale: float = 0.8,
    horizon: int | None = MATCH_HORIZON,
    channel: str | None = None,
) -> dict[str, np.ndarray]:
    """Per-commanding-step delay until the robot does what THAT step asked for.

    Identical criterion to the onset metric — same cone, same speed band, same driver norm
    limit, same channels — differing only in where the clock starts: every commanding step
    rather than the first one of each intervention, so a correction that is resisted for its
    whole duration contributes once per step instead of once.

    A step is CENSORED (excluded, not counted as a failure) when the operator turns more than
    HOLD_COS away from the direction they gave at that step before the criterion is met: the
    reference has gone stale, so the robot is no longer failing to obey it.

    Returns one entry per commanding step, in time order:

        delay_cmd / delay_ach  : timesteps until the arbitrated command / the achieved motion
                                 matched, NaN when it never did inside the horizon
        matched_cmd / _ach     : bool, whether it matched at all
        censored               : bool, the operator turned away before either could resolve
        cmd_strength           : ||human command|| on the primary channel, the stratifier
        policy_cos             : cosine between the policy's action and the operator's;
                                 negative = the policy is pulling against them
        episode                : index into `demos`, for per-episode aggregation
    """
    cols = {k: [] for k in ("delay_cmd", "delay_ach", "matched_cmd", "matched_ach", "censored",
                            "cmd_strength", "policy_cos", "episode")}
    for ep_idx, demo in enumerate(demos):
        sig = _demo_signals(demo, hz, action_scale, channel)
        if sig is None:
            continue
        iv, (t0, t1) = sig["iv"], sig["grip_window"]
        commanding = sig["commanding"] & _window(len(iv), t0, t1)
        parts = sig["channels"]
        # Per-channel unit commands, for the "have they turned away" test.
        hats = {}
        for name, part in parts.items():
            raw = np.asarray(demo["human_actions"], dtype=np.float64)
            block = raw[:len(iv), :3] if name == "linear" else raw[:len(iv), 3:6]
            hats[name] = (_unit(block, part["u_norm"]), part["active"])
        robot = (np.asarray(demo["robot_actions"], dtype=np.float64)[:len(iv)]
                 if demo.get("robot_actions") is not None else None)
        primary = parts[list(parts)[0]]
        r_slice = slice(0, 3) if list(parts)[0] == "linear" else slice(3, 6)

        # None = scan to the end of the intervention, matching the onset metric's default.
        span = len(iv) if horizon is None else int(horizon) + 1
        for t in np.flatnonzero(commanding):
            hit_cmd = hit_ach = None
            stale = False
            for k in range(span):
                tt = t + k
                if tt > t1 or not iv[tt]:
                    break
                # Stale reference: a channel they were driving at t has turned away from it.
                if any(active[t] and active[tt] and float(hat[tt] @ hat[t]) < HOLD_COS
                       for hat, active in hats.values()):
                    stale = True
                    break
                if commanding[tt]:
                    if hit_cmd is None and sig["match_cmd"][tt]:
                        hit_cmd = k
                    if hit_ach is None and sig["match_ach"][tt]:
                        hit_ach = k
                if hit_cmd is not None and hit_ach is not None:
                    break
            cols["delay_cmd"].append(hit_cmd if hit_cmd is not None else np.nan)
            cols["delay_ach"].append(hit_ach if hit_ach is not None else np.nan)
            cols["matched_cmd"].append(hit_cmd is not None)
            cols["matched_ach"].append(hit_ach is not None)
            cols["censored"].append(stale and hit_cmd is None and hit_ach is None)
            cols["cmd_strength"].append(primary["u_norm"][t])
            cols["policy_cos"].append(
                float(robot[t, r_slice] @ hats[list(parts)[0]][0][t])
                / max(np.linalg.norm(robot[t, r_slice]), 1e-12)
                if robot is not None and np.linalg.norm(robot[t, r_slice]) > 1e-9 else np.nan)
            cols["episode"].append(ep_idx)
    dtypes = {"matched_cmd": bool, "matched_ach": bool, "censored": bool, "episode": np.int64}
    return {k: np.asarray(v, dtype=dtypes.get(k, np.float64)) for k, v in cols.items()}


def _window(n: int, t0: int, t1: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[t0:t1 + 1] = True
    return mask


def _matches(vec: np.ndarray, ref: np.ndarray, asked: float) -> bool:
    """Moving along `ref` at `asked` speed within MATCH_TOL — direction AND magnitude."""
    vec = vec / max(float(np.linalg.norm(vec)), 1.0)          # the driver's norm limit
    along = float(vec @ ref)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9 or along / norm < COS_THRESH:
        return False
    return MATCH_TOL is None or (1.0 - MATCH_TOL) * asked <= along <= (1.0 + MATCH_TOL) * asked


def unmatched_by_strength(match: dict, bands=((0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, np.inf)),
                          space: str = "ach", diverging_only: bool = False) -> list[dict]:
    """Share of commanding steps never matched, per push-strength band.

    `diverging_only` keeps the steps where the policy was pulling against the operator, which is
    the condition the similarity term exists to handle.
    """
    keep = np.ones(len(match["cmd_strength"]), dtype=bool)
    if diverging_only:
        keep &= np.isfinite(match["policy_cos"]) & (match["policy_cos"] < 0.0)
    rows = []
    for lo, hi in bands:
        sel = keep & (match["cmd_strength"] >= lo) & (match["cmd_strength"] < hi)
        matched = match[f"matched_{space}"][sel]
        rows.append(dict(band=(lo, hi), n=int(sel.sum()),
                         unmatched=float(1.0 - matched.mean()) if matched.size else np.nan))
    return rows


def episode_unmatched(match: dict, n_demos: int, space: str = "ach",
                      band=None) -> np.ndarray:
    """Per-episode never-matched fraction, optionally restricted to one push-strength band."""
    sel = np.ones(len(match["cmd_strength"]), dtype=bool)
    if band is not None:
        sel &= (match["cmd_strength"] >= band[0]) & (match["cmd_strength"] < band[1])
    out = np.full(n_demos, np.nan)
    for ep in range(n_demos):
        rows = match[f"matched_{space}"][sel & (match["episode"] == ep)]
        if rows.size:
            out[ep] = 1.0 - rows.mean()
    return out
