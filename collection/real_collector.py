"""
RealRobotCollector — generic episode collection loop for physical robots.

A single loop that works for any physical robot via the abstract interfaces.

Sim-specific logic (language embeddings, object randomization, MuJoCo state)
stays in the existing Collector class (collection/collector.py).
"""

from __future__ import annotations
import contextlib
import csv
import select
import sys
import termios
import time
import tty
from collections import deque
from itertools import count

import numpy as np
from scipy.spatial.transform import Rotation as _Rot


@contextlib.contextmanager
def _raw_kbd():
    """Put stdin in cbreak mode so a single keypress ('p', Enter) is delivered
    immediately without waiting for a newline. Restores the previous terminal
    settings on exit (including on exception), so the post-episode prompts —
    which use line-buffered input() — work normally. No-op when stdin is not a
    TTY (piped input, tests), where single-key control isn't available anyway.
    """
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _prompt_success(use_oculus: bool = False):
    """
    Ask the operator to score the episode.
    Returns (outcome, success) where outcome is "saved"|"discarded"|"aborted".
    Flushes stdin first so keystrokes buffered during the episode don't bleed in.

    When use_oculus is True, the prompt also accepts the left controller's
    X (discard) and Y (save-as-success) buttons as an alternative to the
    keyboard. Both input sources are polled from a single loop (rather than
    a background thread) so there's never more than one reader contending
    for stdin once the prompt resolves.
    """
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass

    if not use_oculus:
        while True:
            try:
                val = input("\nEpisode result? (1=success, 0=failure, d=discard, a=abort): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
            if val == "1":
                return "saved", True
            if val == "0":
                return "saved", False
            if val == "d":
                return "discarded", False
            if val == "a":
                return "aborted", False
            print("Please enter 1, 0, d, or a.")

    from xarm.utils.network import get_oculus_reading

    prompt = (
        "\nEpisode result? (1=success, 0=failure, d=discard, a=abort)"
        "  [Oculus left: X=discard, Y=save]: "
    )
    print(prompt, end="", flush=True)
    try:
        while True:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                val = sys.stdin.readline().strip().lower()
                if val == "1":
                    return "saved", True
                if val == "0":
                    return "saved", False
                if val == "d":
                    return "discarded", False
                if val == "a":
                    return "aborted", False
                print(f"Please enter 1, 0, d, or a.\n{prompt}", end="", flush=True)
                continue
            data = get_oculus_reading(timeout=0.05)
            if data:
                if data.get("left_x_button", False):
                    print()
                    return "discarded", False
                if data.get("left_y_button", False):
                    print()
                    return "saved", True
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)


class RealRobotCollector:
    """
    Episode collection loop that composes:
      env        : AbstractRobotEnv
      policy     : AbstractPolicy
      teleop     : AbstractTeleopDevice
      arbitrator : ActionArbitrator (from arbitration/)
      writer     : AbstractDataWriter or None

    Control flow per step:
      1. policy.step(obs)         → robot_action_7d
      2. teleop.get_action(obs)   → (human_action_7d, human_active, is_intervening)
      3. arbitrator.arbitrate(…)  → blended action + metadata
      4. writer.write_step(…)
      5. env.step(blended_action) → obs
    """

    def __init__(
        self,
        env,
        policy,
        teleop,
        arbitrator,
        writer=None,
        control_hz: float = 10.0,
        obs_buffer_len: int | None = None,
        uncertainty_path: str | None = None,
        oculus_buttons: bool = False,
        uncertainty_estimator=None,
        uncertainty_bounds=None,
        uncertainty_debug_interval: int = 10,
        eval_mode: bool = False,
        debug_blend_log: str | None = None,
        intervene_lead_m: float | None = 0.005,
    ):
        self.env = env
        self.policy = policy
        self.teleop = teleop
        self.arbitrator = arbitrator
        self.writer = writer
        self.control_hz = control_hz
        self.uncertainty_path = uncertainty_path
        self.oculus_buttons = oculus_buttons
        self.uncertainty_estimator = uncertainty_estimator
        self.uncertainty_bounds = uncertainty_bounds
        self.uncertainty_debug_interval = uncertainty_debug_interval
        self.eval_mode = eval_mode
        self.debug_blend_log = debug_blend_log
        self.intervene_lead_m = intervene_lead_m
        self._episode_id = 0
        self.last_episode_health: dict = {}

        self.obs_buffer = (
            deque(maxlen=obs_buffer_len) if obs_buffer_len is not None else None
        )

    # ------------------------------------------------------------------
    _DBG_GROUPS = ("xy", "z", "ori")
    _DBG_GROUP_FIELDS = ("mag", "sim", "conf", "alpha_sim", "alpha_conf",
                         "alpha_target", "alpha_final")

    def _blend_log_fieldnames(self) -> list[str]:
        base = [
            "step", "t", "control_mode", "human_active", "intervening",
            "grip_held", "on_resume", "disengaged", "is_first_intervention_step",
            "human_inactive_count", "gate_w", "gate_env", "target_held",
            "arm_x", "arm_y", "arm_z", "quat_x", "quat_y", "quat_z", "quat_w",
            "gripper_pos",
        ]
        base += [f"h{i}" for i in range(7)]   # raw human action (pre-blend)
        base += [f"r{i}" for i in range(7)]   # raw policy/robot action
        base += [f"e{i}" for i in range(7)]   # executed (blended) action
        for g in self._DBG_GROUPS:
            base += [f"{g}_{f}" for f in self._DBG_GROUP_FIELDS]
        return base

    def _open_blend_log(self):
        """Open the optional per-step blend log (CSV) for this episode. Returns (file, DictWriter)."""
        from pathlib import Path
        out_dir = Path(self.debug_blend_log)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"blend_debug_ep{self._episode_id:03d}_{ts}.csv"
        f = open(path, "w", newline="")
        w = csv.DictWriter(f, fieldnames=self._blend_log_fieldnames())
        w.writeheader()
        print(f"[collect] blend debug log → {path}")
        return f, w

    def _write_blend_row(self, writer, *, step, t, obs, human_action, robot_action,
                         exec_action, control_mode, human_active, intervening,
                         grip_held, on_resume, arb_metadata):
        pos  = np.asarray(obs.get("arm_pos",  np.zeros(3))).ravel()
        quat = np.asarray(obs.get("arm_quat", np.zeros(4))).ravel()
        grip = float(np.asarray(obs.get("gripper_pos", [0.0])).ravel()[0])
        row = {
            "step": step, "t": round(t, 4), "control_mode": control_mode,
            "human_active": int(bool(human_active)), "intervening": int(bool(intervening)),
            "grip_held": int(bool(grip_held)), "on_resume": int(bool(on_resume)),
            "disengaged": int(bool(arb_metadata.get("disengaged", False))),
            "is_first_intervention_step": int(bool(arb_metadata.get("is_first_intervention_step", False))),
            "human_inactive_count": arb_metadata.get("human_inactive_count", ""),
            "gate_w": arb_metadata.get("gate_w", ""),
            "gate_env": arb_metadata.get("gate_env", ""),
            "target_held": int(bool(arb_metadata.get("target_held", False))),
            "arm_x": pos[0] if pos.size > 0 else "",
            "arm_y": pos[1] if pos.size > 1 else "",
            "arm_z": pos[2] if pos.size > 2 else "",
            "quat_x": quat[0] if quat.size > 0 else "",
            "quat_y": quat[1] if quat.size > 1 else "",
            "quat_z": quat[2] if quat.size > 2 else "",
            "quat_w": quat[3] if quat.size > 3 else "",
            "gripper_pos": grip,
        }
        for i in range(7):
            row[f"h{i}"] = float(human_action[i])
            row[f"r{i}"] = float(robot_action[i])
            row[f"e{i}"] = float(exec_action[i])
        groups = (arb_metadata or {}).get("groups", {}) or {}
        for g in self._DBG_GROUPS:
            gd = groups.get(g, {})
            for fld in self._DBG_GROUP_FIELDS:
                row[f"{g}_{fld}"] = gd.get(fld, "")
        writer.writerow(row)

    # ------------------------------------------------------------------
    def _pause(self, prev_action) -> float | None:
        """Block the control loop until the operator resumes or ends the episode.

        Entered when the operator presses 'p'. While paused the arm is commanded
        to zero velocity every tick (gripper held at its last value) so it stays
        put and the servo stays fed, and no steps are logged — so the paused
        wall-time enters neither the recorded trajectory nor any timing. Assumes
        stdin is already in cbreak mode (set by the run_episode loop) so single
        keypresses arrive without Enter. Returns the paused duration in seconds
        when the operator presses 'p' again to resume, or None if they press
        Enter to end the episode from the pause.
        """
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass
        stop_action = np.zeros(7, dtype=np.float32)
        stop_action[6] = prev_action[6]      # hold the current gripper position
        # Command zero velocity once; the driver holds it, and not polling keeps paused frames out of health stats.
        self.env.step(stop_action)
        pause_start = time.time()
        print("[collect] PAUSED — press 'p' to resume, Enter to end episode.")
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1).lower()
                if ch == "p":
                    pause_dt = time.time() - pause_start
                    print(f"[collect] RESUMED after {pause_dt:.1f}s paused.")
                    return pause_dt
                if ch in ("\n", "\r"):
                    # Enter ends the episode from the pause.
                    return None
                # Any other key is ignored; stay paused.

    # ------------------------------------------------------------------
    def run_episode(self, max_steps: int | None = None, num_restarts: int = 0) -> tuple[str, int]:
        """
        Run one episode.

        Returns (outcome, n_steps) where outcome is "saved" | "discarded" | "aborted"
        and n_steps is the number of control steps executed.
        """
        self.policy.reset()
        self.arbitrator.reset()
        self.teleop.reset()
        _arb_est = getattr(self.arbitrator, "uncertainty_estimator", None)
        if hasattr(_arb_est, "reset"):
            _arb_est.reset()
        if hasattr(_arb_est, "reset_stats"):
            _arb_est.reset_stats()

        obs = self.env.reset()
        self.policy.start_episode(obs)

        # Seed the gripper hold from obs so the first step doesn't open a gripper left closed.
        _init_grip = float(obs.get("gripper_pos", np.zeros(1))[0])

        if self.obs_buffer is not None:
            self.obs_buffer.clear()
            for _ in range(self.obs_buffer.maxlen):
                self.obs_buffer.append(obs)

        if hasattr(self.arbitrator, "calibrate"):
            self.arbitrator.calibrate(
                env=self.env,
                obs_buffer=self.obs_buffer,
                lang_emb=None,
                policy=self.policy,
                uncertainty_path=self.uncertainty_path,
            )

        if self.writer is not None:
            self.writer.start_episode()

        prev_action = np.zeros(7, dtype=np.float32)
        prev_action[6] = _init_grip   # hold the real gripper position from the start
        _last_robot_action = None
        _was_intervening = False
        _steps_since_interv = 10 ** 9   # no tether until the first intervention
        start_time = time.time()
        _paused_accum = 0.0   # total operator-paused wall-time this episode
        _gap_count = 0

        _dbg_file = _dbg_writer = None
        if self.debug_blend_log:
            _dbg_file, _dbg_writer = self._open_blend_log()

        # ── Frame-time instrumentation ────────────────────────────────────────
        # Per-frame work time by stage, since average Hz hides stalls; overruns >1.5x period warn.
        _nominal = 1.0 / self.control_hz
        _stage_names = ("obs", "policy", "unc", "teleop", "arb", "write", "env")
        _stage_sums = dict.fromkeys(_stage_names, 0.0)
        _frame_dts = []      # per-frame WORK time (obs→env.step, excludes sleep)
        _frame_starts = []   # per-frame start wall clock → true achieved period
        _frame_interv = []   # per-frame intervening flag → per-mode timing breakout
        _overruns = 0
        _last_overrun_warn = 0.0

        def _timing_summary():
            if len(_frame_starts) < 2:
                return {}
            fd = np.asarray(_frame_dts)
            n = len(fd)
            periods = np.diff(_frame_starts)   # actual step-to-step time, sleep included
            slow_n = int(np.sum(periods > 1.5 * _nominal))
            stage_str = "  ".join(
                f"{k}={_stage_sums[k] / n * 1000:.1f}" for k in _stage_names
                if _stage_sums[k] / n > 0.001
            )
            print(f"[timing] achieved: median {1.0/np.median(periods):.1f} Hz  "
                  f"(target {self.control_hz:.0f})  p95 period={np.percentile(periods,95)*1000:.0f}ms  "
                  f"max={periods.max()*1000:.0f}ms  "
                  f"frames slower than {1.0/(1.5*_nominal):.0f} Hz: {slow_n}/{len(periods)}")
            # Per-mode breakout: an overall median hides intervention-only slowdowns.
            iv = np.asarray(_frame_interv[:len(periods)], dtype=bool)
            if iv.any() and (~iv).any():
                for name, sel in (("intervention", iv), ("autonomous", ~iv)):
                    p = periods[sel]
                    print(f"[timing]   {name:12s}: median {1.0/np.median(p):.1f} Hz  "
                          f"p95 period={np.percentile(p,95)*1000:.0f}ms  "
                          f"slow frames: {int(np.sum(p > 1.5*_nominal))}/{len(p)}")
            # Async uncertainty results come from an older observation; flag the episode if they lagged.
            est = getattr(self.arbitrator, "uncertainty_estimator", None)
            if hasattr(est, "stats"):
                s = est.stats()
                if s["n"] > 0:
                    flag = (f"  ** STALE: {s['stale_frac']:.0%} of reads older than "
                            f"{s['stale_warn_s']*1000:.0f}ms — don't trust conf this episode"
                            if s["stale_frac"] > 0.05 else "")
                    print(f"[timing] uncertainty age: median={s['median_s']*1000:.0f}ms  "
                          f"p95={s['p95_s']*1000:.0f}ms  max={s['max_s']*1000:.0f}ms  "
                          f"no-result frames={s['none_returns']}{flag}")
            print(f"[timing] work/frame: mean={fd.mean()*1000:.1f}ms  "
                  f"p95={np.percentile(fd, 95)*1000:.0f}ms  max={fd.max()*1000:.0f}ms  "
                  f"overruns(>{1.5*_nominal*1000:.0f}ms)={_overruns} ({_overruns/n:.0%})  "
                  f"| stage means (ms): {stage_str}")
            return {
                "achieved_hz_median": float(1.0 / np.median(periods)),
                "period_p95_ms":      float(np.percentile(periods, 95) * 1000),
                "period_max_ms":      float(periods.max() * 1000),
                "slow_frames":        int(slow_n),
                "n_frames":           int(len(periods)),
                "overruns":           int(_overruns),
                "episode_wall_s":     float(_frame_starts[-1] - _frame_starts[0]),
            }
        _stale_break = False

        def _collect_health(summary):
            health = dict(summary or {})
            if hasattr(self.env, "episode_health"):
                health.update(self.env.episode_health())
            health["ended_on_stale_camera"] = bool(_stale_break)
            self.last_episode_health = health
            return health

        _prev_rot_m    = None
        _prev_pos      = None
        _prev_qpos     = None
        _prev_raw_qpos = None   # raw hardware angles (wrapped) for delta computation
        _unwrapped_qpos: np.ndarray | None = None  # cumulative unwrapped joint angles

        # A frozen orbbec1 feed would corrupt recorded observations, so it ends the episode.
        _orbbec1_cam = None
        for _cam, _name in zip(getattr(self.env, "_cameras", []),
                               getattr(self.env, "_camera_names", [])):
            if _name == "orbbec1":
                _orbbec1_cam = _cam
                break

        print("Episode running. Press Enter to end episode, 'p' to pause/resume.")

        with _raw_kbd():
            for step_idx in count():
                if max_steps is not None and step_idx >= max_steps:
                    break

                # ── Keyboard: 'p' pauses/resumes (single keypress); Enter ends ───
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1).lower()
                    if ch == "p":
                        pause_dt = self._pause(prev_action)
                        if pause_dt is None:
                            print("Ending episode from pause.")
                            break
                        # Shift episode clocks by the pause so scheduling, timestamps and task time exclude it.
                        start_time    += pause_dt
                        _paused_accum += pause_dt
                        if self.writer is not None and hasattr(self.writer, "notify_pause"):
                            self.writer.notify_pause(pause_dt)
                        # Re-base the policy's virtual target so the first post-resume action doesn't lurch.
                        if hasattr(self.policy, "on_resume"):
                            self.policy.on_resume(self.env.get_obs())
                    elif ch in ("\n", "\r"):
                        print("Enter pressed — ending episode.")
                        break
                    # Any other key is ignored (no accidental episode end).
                if getattr(self.teleop, "episode_done", False):
                    print("Controller button — ending episode.")
                    break

                # ── Timing ───────────────────────────────────────────────────────
                target_t = start_time + step_idx / self.control_hz
                sleep_dt = target_t - time.time()
                if 0 < sleep_dt < 2.0 / self.control_hz:
                    time.sleep(sleep_dt)

                # ── Observation ──────────────────────────────────────────────────
                _f0 = _tmark = time.time()
                # Episode clock excluding paused time, so logged timestamps have no gap.
                _step_t = _f0 - _paused_accum

                def _stage(name):
                    nonlocal _tmark
                    now = time.time()
                    _stage_sums[name] += now - _tmark
                    _tmark = now

                obs = self.env.get_obs()

                # Stale orbbec1 feed ends the episode as if Enter were pressed.
                if (_orbbec1_cam is not None
                        and hasattr(_orbbec1_cam, "is_stale")
                        and _orbbec1_cam.is_stale(max_age_s=0.5)):
                    print("[collect] orbbec1 camera stale — ending episode.")
                    _stale_break = True
                    break

                # Unwrap arm_qpos to be continuous; normalise to [-π, π] at episode start.
                raw_qpos = obs["arm_qpos"].copy()
                if _prev_raw_qpos is None:
                    _unwrapped_qpos = (((raw_qpos + np.pi) % (2 * np.pi)) - np.pi).astype(np.float64)
                else:
                    qpos_delta = (((raw_qpos - _prev_raw_qpos + np.pi) % (2 * np.pi)) - np.pi)
                    _unwrapped_qpos = _unwrapped_qpos + qpos_delta
                _prev_raw_qpos = raw_qpos
                obs["arm_qpos"] = _unwrapped_qpos.astype(np.float32)

                # Compute EEF and joint deltas
                curr_rot_m = _Rot.from_quat(obs["arm_quat"]).as_matrix()
                if _prev_rot_m is None:
                    obs["eef_pos_delta"]   = np.zeros(3,  dtype=np.float32)
                    obs["eef_rot6d_delta"] = np.array([1,0,0, 0,1,0], dtype=np.float32)
                    obs["joint_delta"]     = np.zeros_like(obs["arm_qpos"])
                else:
                    obs["eef_pos_delta"]   = (_prev_rot_m.T @ (obs["arm_pos"] - _prev_pos)).astype(np.float32)
                    obs["eef_rot6d_delta"] = (_prev_rot_m.T @ curr_rot_m)[:2, :].flatten().astype(np.float32)
                    # Both _prev_qpos and arm_qpos are now unwrapped; delta is already correct.
                    obs["joint_delta"] = (obs["arm_qpos"] - _prev_qpos).astype(np.float32)
                _prev_rot_m = curr_rot_m
                _prev_pos   = obs["arm_pos"].copy()
                _prev_qpos  = obs["arm_qpos"].copy()

                if self.obs_buffer is not None:
                    self.obs_buffer.append(obs)
                    obs_input = {
                        k: np.stack([o[k] for o in self.obs_buffer], axis=0)
                        for k in obs
                    }
                else:
                    obs_input = obs

                # ── Robot policy ─────────────────────────────────────────────────
                # Tether the policy's virtual target to the actual pose during and ~1 s after intervention to avoid wind-up.
                _stage("obs")
                if (_steps_since_interv <= self.control_hz
                        and self.intervene_lead_m is not None
                        and hasattr(self.policy, "constrain_lead")):
                    self.policy.constrain_lead(obs, self.intervene_lead_m)

                robot_action = self.policy.step(obs_input)
                _stage("policy")

                # ── Uncertainty compute (every step) + print (every N steps) ─────
                _unc_raw  = None
                _unc_norm = None
                if self.uncertainty_estimator is not None and robot_action is not None:
                    try:
                        _unc_raw = self.uncertainty_estimator.compute(
                            self.obs_buffer, robot_action, self.policy
                        )
                        if _unc_raw is not None and self.uncertainty_bounds is not None:
                            l_min, l_max = self.uncertainty_bounds
                            span = np.maximum(l_max - l_min, 1e-8)
                            _unc_norm = np.clip((_unc_raw - l_min) / span, 0.0, 1.0)
                            if step_idx % self.uncertainty_debug_interval == 0:
                                raw_str  = " ".join(f"{v:.4f}" for v in _unc_raw)
                                norm_str = " ".join(f"{v:.3f}" for v in _unc_norm)
                                print(
                                    f"[unc step={step_idx:4d}] "
                                    f"raw=[{raw_str}]  "
                                    f"norm=[{norm_str}]  "
                                    f"mean={_unc_norm.mean():.3f}"
                                )
                    except Exception as _e:
                        print(f"[unc step={step_idx}] estimator error: {_e}")
                _stage("unc")

                # Handle inference gap (policy not ready this step)
                if robot_action is None:
                    _gap_count += 1
                    if _last_robot_action is not None:
                        robot_action = _last_robot_action
                    else:
                        robot_action = np.zeros(7, dtype=np.float32)
                else:
                    robot_action = np.clip(robot_action, -1.0, 1.0)
                    _last_robot_action = robot_action.copy()
                    if _gap_count > 0:
                        _gap_count = 0

                # ── Human input ──────────────────────────────────────────────────
                human_action, human_active, is_intervening = self.teleop.get_action(
                    obs, prev_action
                )
                _stage("teleop")

                if human_action is None:
                    # Teleop device signalled episode abort (e.g. hold both buttons)
                    print("Teleop abort signal — saving as aborted.")
                    if self.writer is not None:
                        self.writer.abort_episode()
                    if _dbg_file is not None:
                        _dbg_file.close()
                    _collect_health(_timing_summary())
                    self._episode_id += 1
                    return "aborted", step_idx

                # ── Arbitrate ────────────────────────────────────────────────────
                arb = self.arbitrator.arbitrate(
                    obs_buffer=self.obs_buffer,
                    robot_action=robot_action,
                    human_action=human_action,
                    human_active=human_active,
                    intervening=is_intervening,
                    policy=self.policy,
                )

                _stage("arb")
                exec_action   = arb["action"].copy()
                control_mode  = arb["control_mode"]
                training      = arb.get("training", False)
                intervening   = arb.get("intervening", is_intervening)
                alphas        = arb.get("alphas", None)
                arb_metadata  = arb.get("metadata", {})

                if step_idx % 30 == 0:
                    grip_str = "grip=held" if getattr(self.teleop, "grip_held", True) else "grip=OFF"
                    print(
                        f"[ctrl step={step_idx:4d}] {control_mode:7s}  "
                        f"human_active={str(human_active):5s}  "
                        f"intervening={str(intervening):5s}  {grip_str}"
                    )

                grip_held   = getattr(self.teleop, "grip_held", True)
                full_teleop = type(self.policy).__name__ == "NullPolicy"

                # On takeover, re-base the teleop gripper integrator to the hardware position to avoid a snap.
                if intervening and not _was_intervening:
                    grip_obs = obs.get("gripper_pos")
                    if grip_obs is not None and hasattr(self.teleop, "_gripper_pos"):
                        hw = float(grip_obs[0] if hasattr(grip_obs, "__len__") else grip_obs)
                        self.teleop._gripper_pos = hw
                        human_action = human_action.copy()
                        human_action[6] = hw

                # On handback, reset the policy's virtual integrator to the actual arm pose.
                _on_resume_fired = bool(_was_intervening and not intervening)
                if _on_resume_fired:
                    if hasattr(self.policy, "on_resume"):
                        self.policy.on_resume(obs)
                _was_intervening = intervening
                _steps_since_interv = 0 if intervening else _steps_since_interv + 1

                # Gripper DOF follows whoever controls the arm; dead-man release holds everything (not in eval mode).
                if not self.eval_mode:
                    if not grip_held:
                        # Grip released: neither human nor policy moves the robot.
                        exec_action = np.zeros(7, dtype=np.float32)
                        exec_action[6] = prev_action[6]      # hold gripper in place
                    elif intervening:
                        exec_action[6] = human_action[6]     # human owns the gripper
                    elif full_teleop:
                        exec_action[6] = prev_action[6]      # no policy gripper; hold
                    else:
                        exec_action[6] = robot_action[6]     # policy owns the gripper

                # ── Write step ───────────────────────────────────────────────────
                if self.writer is not None:
                    self.writer.log_step(
                        obs,
                        executed_action=exec_action,
                        robot_action=robot_action,
                        human_action=human_action,
                        control_mode=control_mode,
                        human_active=human_active,
                        intervening=intervening,
                        grip_held=grip_held,
                        training=training,
                        alphas=alphas,
                        uncertainty_raw=_unc_raw,
                        uncertainty_norm=_unc_norm,
                        blend_terms=arb.get("blend_terms"),
                        blend_meta=arb_metadata,
                        step_time=_step_t,
                    )

                # ── Optional per-step blend log (CSV) ──────────────────────────────
                if _dbg_writer is not None:
                    try:
                        self._write_blend_row(
                            _dbg_writer, step=step_idx, t=time.time() - start_time,
                            obs=obs, human_action=human_action, robot_action=robot_action,
                            exec_action=exec_action, control_mode=control_mode,
                            human_active=human_active, intervening=intervening,
                            grip_held=grip_held, on_resume=_on_resume_fired,
                            arb_metadata=arb_metadata,
                        )
                    except Exception as _e:
                        print(f"[collect] blend debug log error at step {step_idx}: {_e}")

                _stage("write")

                # ── Execute ──────────────────────────────────────────────────────
                obs, _reward, done, _info = self.env.step(exec_action)
                _stage("env")
                prev_action = exec_action

                # ── Frame-time watchdog ──────────────────────────────────────────
                _frame_dt = time.time() - _f0
                _frame_dts.append(_frame_dt)
                _frame_starts.append(_step_t)   # episode clock: excludes paused time
                _frame_interv.append(bool(intervening))
                if _frame_dt > 1.5 * _nominal:
                    _overruns += 1
                    _now = time.time()
                    if _now - _last_overrun_warn > 1.0:   # throttle: ≤1 print/s
                        _last_overrun_warn = _now
                        _worst = sorted(_stage_names,
                                        key=lambda k: _stage_sums[k], reverse=True)[:3]
                        print(f"[timing] step {step_idx}: frame took {_frame_dt*1000:.0f}ms "
                              f"(budget {_nominal*1000:.0f}ms, {control_mode}) — "
                              f"cumulative worst stages: "
                              + ", ".join(f"{k}={_stage_sums[k]/max(len(_frame_dts),1)*1000:.1f}ms/frame"
                                          for k in _worst))

                if done:
                    break

        if _dbg_file is not None:
            _dbg_file.close()

        _health = _collect_health(_timing_summary())

        # Zero velocity so the robot stops moving while we wait for input.
        # Preserve the last gripper value to avoid snapping it open.
        stop_action = np.zeros(7, dtype=np.float32)
        stop_action[6] = prev_action[6]
        self.env.step(stop_action)

        # ── Prompt operator for outcome ───────────────────────────────────────
        outcome, success = _prompt_success(use_oculus=self.oculus_buttons)

        if self.writer is not None:
            if outcome == "saved":
                self.writer.log_episode(success=success, num_restarts=num_restarts,
                                        health=_health)
            elif outcome == "aborted":
                self.writer.abort_episode()
            else:
                self.writer.discard_episode()

        self._episode_id += 1
        return outcome, step_idx
