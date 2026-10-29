import numpy as np
import time
from collections import deque
from collection.multiview import MultiViewRenderer
from collection.randomize_pos import randomize_obj_pos, randomize_for_task
from collection.teleop.teleop import Teleop
import torch
import random
import robomimic.utils.lang_utils as LangUtils

class Collector:
    def __init__(self, env, policy, task, torch_device, human, writer, arbitrator, keep_if_unsuccessful, seed, render=True, use_sequence=False, seq_length=None, uncertainty_path=None):
        self.env = env
        self.policy = policy
        self.task = task
        self.torch_device = torch_device
        self.human = human  # teleop device wrapper
        self.writer = writer
        self.arbitrator = arbitrator
        self.keep_if_unsuccessful = keep_if_unsuccessful
        self.viewer = MultiViewRenderer(env)
        self.render = render
        self.seed = seed
        if use_sequence:
            self.obs_buffer = deque(maxlen=seq_length)
        else:
            self.obs_buffer = None
        self.cached_state_dict = None
        self.uncertainty_path = uncertainty_path

    def rollout(self, max_horizon=500, create_grp=False, state_dict=None):
        self.seed += 1

        # Seed global RNGs
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Seed Robosuite/Robocasa's internal RandomState specifically
        if hasattr(self.env, "env") and hasattr(self.env.env, "np_random"):
            self.env.env.np_random = np.random.RandomState(self.seed)

        if state_dict is None and self.cached_state_dict is not None:
            state_dict = self.cached_state_dict

        if state_dict is None:
            obs = self.env.reset()
            self.cached_state_dict = self.env.get_state()
        else:
            obs = self.env.reset_to(state_dict)

        # For deterministic action playback (comes directly from robomimic run_trained_agent)
        if state_dict is not None:
            print("USING SEED: ", self.seed)
            randomize_for_task(self.env, self.task, seed=self.seed) # per-task object/base jitter

        lang_emb = self.get_lang_emb(self.env, device=self.torch_device)
        if lang_emb is not None:
            obs["lang_emb"] = lang_emb
        if self.obs_buffer is not None:
            for _ in range(self.obs_buffer.maxlen):
                self.obs_buffer.append(obs)
        
        self.policy.start_episode(lang=lang_emb)
        self.policy.policy.set_eval()

        self.arbitrator.calibrate(env=self.env, obs_buffer=self.obs_buffer, lang_emb=lang_emb, policy=self.policy, uncertainty_path=self.uncertainty_path)
        
        self.writer.start_episode(create_grp=create_grp)
        self.writer.set_language_embedding(lang_emb)

        # Initialize success counter
        task_completion_hold_count = -1

        t = 0
        success = False
        self.human.reset()
        control_freq = 20.0 
        playback_speed = 0.4  # speed compared to real-time
        target_loop_duration = (1.0 / control_freq) / playback_speed
        prev_exec_action = None
        _rt_pol=[]; _rt_arb=[]; _rt_step=[]; _rt_rend=[]; _rt_work=[]   # rolling live-print timers
        # Per-episode timing (frame starts, per-mode flags, HID age) recorded in the episode meta.
        _ep_starts=[]; _ep_work=[]; _ep_interv=[]
        _ep_stage={"pol":0.0,"arb":0.0,"step":0.0,"rend":0.0}
        _ep_input_age=[]  # HID age at consumption, active frames only (idle frames have no packets)
        _device = getattr(self.human, "device", None)
        _sched_t0 = None; _sched_i = 0   # absolute step schedule

        def _timing_summary():
            """Print achieved rate / input lag for the episode and return it for the meta JSON."""
            if len(_ep_starts) < 2:
                return {}
            periods = np.diff(_ep_starts)
            work = np.asarray(_ep_work); n = len(work)
            slow_n = int(np.sum(periods > 1.5 * target_loop_duration))
            over_n = int(np.sum(work > target_loop_duration))
            print(f"[timing] achieved: median {1.0/np.median(periods):.1f} Hz (target {1.0/target_loop_duration:.1f}) "
                  f"p95 period={np.percentile(periods,95)*1000:.0f}ms max={periods.max()*1000:.0f}ms  "
                  f"frames slower than {1.0/(1.5*target_loop_duration):.1f} Hz: {slow_n}/{len(periods)}", flush=True)
            # Per-mode breakout: an overall median hides intervention-only slowdowns.
            iv = np.asarray(_ep_interv[:len(periods)], dtype=bool)
            if iv.any() and (~iv).any():
                for name, sel in (("intervention", iv), ("autonomous", ~iv)):
                    p = periods[sel]
                    print(f"[timing]   {name:12s}: median {1.0/np.median(p):.1f} Hz  "
                          f"p95 period={np.percentile(p,95)*1000:.0f}ms  "
                          f"slow frames: {int(np.sum(p > 1.5*target_loop_duration))}/{len(p)}", flush=True)
            print(f"[timing] work/frame: mean={work.mean()*1000:.1f}ms p95={np.percentile(work,95)*1000:.0f}ms "
                  f"max={work.max()*1000:.0f}ms overruns(>{target_loop_duration*1000:.0f}ms)={over_n} ({over_n/n:.0%})  "
                  f"| stage means (ms): pol={_ep_stage['pol']/n*1000:.1f} arb={_ep_stage['arb']/n*1000:.1f} "
                  f"step={_ep_stage['step']/n*1000:.1f} rend={_ep_stage['rend']/n*1000:.1f}", flush=True)
            out = {
                "achieved_hz_median": float(1.0/np.median(periods)),
                "target_hz":          float(1.0/target_loop_duration),
                "period_p95_ms":      float(np.percentile(periods,95)*1000),
                "period_max_ms":      float(periods.max()*1000),
                "slow_frames":        slow_n,
                "n_frames":           int(n),
                "overruns":           over_n,
                "work_mean_ms":       float(work.mean()*1000),
                "work_p95_ms":        float(np.percentile(work,95)*1000),
                "work_max_ms":        float(work.max()*1000),
                "episode_wall_s":     float(_ep_starts[-1]-_ep_starts[0]),
            }
            # Input age = wall time between the newest HID packet and the step that consumed it.
            if len(_ep_input_age) > 0:
                age = np.asarray(_ep_input_age) * 1000
                print(f"[timing] input age (active frames): median={np.median(age):.0f}ms "
                      f"p95={np.percentile(age,95):.0f}ms max={age.max():.0f}ms  n={len(age)}", flush=True)
                out.update({
                    "input_age_median_ms":  float(np.median(age)),
                    "input_age_p95_ms":     float(np.percentile(age,95)),
                    "input_age_max_ms":     float(age.max()),
                    "input_active_frames":  int(len(age)),
                })
            if hasattr(_device, "input_stats"):
                s = _device.input_stats()
                out.update({
                    "input_packets":         int(s["packets"]),
                    "input_stale_trips":     int(s["stale_trips"]),
                    "input_stale_timeout_s": float(s["stale_timeout_s"]),
                })
                if s["stale_trips"] > 0:
                    print(f"[timing] ** HID went silent {s['stale_trips']}x (>{s['stale_timeout_s']*1000:.0f}ms) and input "
                          f"was zeroed — check the input device connection before trusting this episode", flush=True)
            return out

        def _finish_timing():
            if hasattr(self.writer, "set_episode_health"):
                self.writer.set_episode_health(_timing_summary())

        while True:
            loop_start_time = time.time() # Start stopwatch
            _ep_starts.append(loop_start_time)

            if self.obs_buffer is not None:
                obs_seq = {k: np.stack([o[k] for o in self.obs_buffer], axis=0) for k in obs.keys()}
                robot_action = self.policy(obs_seq)
            else:
                robot_action = self.policy(obs)

            robot_action = np.clip(robot_action, -1.0, 1.0) # ensure action is in valid range, get rid of floating point errors

            # Per-step policy uncertainty (raw + normalized) for logging parity with real_collector.
            _unc_raw = None; _unc_norm = None
            _est = getattr(self.arbitrator, "uncertainty_estimator", None)
            if _est is not None and robot_action is not None:
                try:
                    _unc_raw = _est.compute(obs_buffer=self.obs_buffer, robot_action=robot_action, policy=self.policy)
                    _lmin = getattr(self.arbitrator, "L_MIN", None); _lmax = getattr(self.arbitrator, "L_MAX", None)
                    if _unc_raw is not None and _lmin is not None and _lmax is not None:
                        _span = np.maximum(np.asarray(_lmax) - np.asarray(_lmin), 1e-8)
                        _unc_norm = np.clip((np.asarray(_unc_raw) - np.asarray(_lmin)) / _span, 0.0, 1.0)
                except Exception:
                    _unc_raw = None; _unc_norm = None
            _t1 = time.time()

            human_action, action_dict, intervening = self.human.get_action(obs=obs, robot_action=robot_action, prev_action=prev_exec_action)
            if human_action is None:
                print("Reset detected — restarting.")
                self.human.reset() # reset teleop device state
                _finish_timing()
                self.writer.abort_episode()
                self.seed -= 1 # repeat same environment for next rollout since this one is discarded
                return -1
            
            # Hard-switch takeover invalidates the robot's queued chunk; blends co-execute so it stays valid.
            if intervening and getattr(self.arbitrator, "_invalidates_robot_plan", False) and hasattr(self.policy.policy, "action_queue"):
                self.policy.policy.action_queue.clear()

            human_active = self.human.has_nonzero_action(action_dict)
            gripper_active = self.human.has_gripper_action(action_dict)

            if human_active and hasattr(_device, "input_age"):
                _age = _device.input_age()
                if _age is not None:
                    _ep_input_age.append(_age)

            _ta = time.time()
            arb = self.arbitrator.arbitrate(
                obs_buffer=self.obs_buffer,
                robot_action=robot_action,
                human_action=human_action,
                human_active=human_active,
                gripper_active=gripper_active,
                intervening=intervening,
                policy=self.policy,
                info={"action_dict": action_dict},
            )
            _t2 = time.time()

            action = arb["action"]
            control_mode = arb["control_mode"]
            training = arb["training"]
            intervening = arb["intervening"]
            metadata = arb.get("metadata", {})
            alphas = arb.get("alphas", None)
            blend_terms = arb.get("blend_terms", None)

            sim_state = self.env.env.sim.get_state()
            # qacc is from mj_forward at the end of the previous step (state t, ctrl t-1).
            # This one-step lag is intentional: at a transition, qacc[t] = robot response,
            # qacc[t+1] = human response, so the difference captures the handoff jerk cleanly.
            joint_pos = sim_state.qpos.copy()
            joint_vel = sim_state.qvel.copy()
            joint_acc = self.env.env.sim.data.qacc.copy()

            # Observed base-frame eef pose + gripper, used by trim-zeros.
            _ep = obs["robot0_base_to_eef_pos"] if "robot0_base_to_eef_pos" in obs else obs.get("robot0_eef_pos")
            _eq = obs["robot0_base_to_eef_quat"] if "robot0_base_to_eef_quat" in obs else obs.get("robot0_eef_quat")
            _gp = obs.get("robot0_gripper_qpos")
            _eef_pos = np.asarray(_ep).copy() if _ep is not None else None
            _eef_quat = np.asarray(_eq).copy() if _eq is not None else None
            _gripper_pos = np.asarray(_gp).copy() if _gp is not None else None

            self.writer.log_step(
                executed_action=action, # normal scale action
                human_action=human_action if human_active else None,
                robot_action=robot_action,
                control_mode=control_mode,
                gripper_active=gripper_active,
                intervening=intervening,
                training=training,
                alphas=alphas,
                metadata=metadata,
                blend_terms=blend_terms,
                uncertainty_raw=_unc_raw,
                uncertainty_norm=_unc_norm,
                sim_state=sim_state.flatten(),
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                joint_acc=joint_acc,
                eef_pos=_eef_pos,
                eef_quat=_eef_quat,
                gripper_pos=_gripper_pos,
            )

            obs, _, env_done, _ = self.env.step(action)
            _t3 = time.time()
            prev_exec_action = action
            if lang_emb is not None:
                obs["lang_emb"] = lang_emb
            if self.obs_buffer is not None:
                self.obs_buffer.append(obs)

            if self.render:
                self.viewer.render()
            _t4 = time.time()

            # Print achieved loop rate vs target every 40 steps.
            _rt_pol.append(_t1-loop_start_time); _rt_arb.append(_t2-_ta); _rt_step.append(_t3-_t2); _rt_rend.append(_t4-_t3); _rt_work.append(_t4-loop_start_time)
            _ep_work.append(_t4-loop_start_time); _ep_interv.append(bool(intervening))
            _ep_stage["pol"] += _t1-loop_start_time; _ep_stage["arb"] += _t2-_ta; _ep_stage["step"] += _t3-_t2; _ep_stage["rend"] += _t4-_t3
            if len(_rt_work) >= 40:
                _n=len(_rt_work); _over=sum(w>target_loop_duration for w in _rt_work); _m=lambda a:1000*np.mean(a)
                print(f"[rate] work {_m(_rt_work):.0f}ms (max {1000*max(_rt_work):.0f}) target {1000*target_loop_duration:.0f}ms  overrun {_over}/{_n}  eff {1.0/max(np.mean(_rt_work),target_loop_duration):.1f}Hz | pol {_m(_rt_pol):.0f} arb {_m(_rt_arb):.0f} step {_m(_rt_step):.0f} rend {_m(_rt_rend):.0f}", flush=True)
                _rt_pol.clear(); _rt_arb.clear(); _rt_step.clear(); _rt_rend.clear(); _rt_work.clear()

            # Slow the loop so the human can intervene in time; absolute schedule avoids drift below target.
            if type(self.human) is Teleop:
                if _sched_t0 is None:
                    _sched_t0 = loop_start_time
                _sched_i += 1
                _sleep = (_sched_t0 + _sched_i * target_loop_duration) - time.time()
                if _sleep > 0:
                    time.sleep(min(_sleep, 2 * target_loop_duration))
                elif _sleep < -target_loop_duration:
                    _sched_t0 = time.time() - _sched_i * target_loop_duration  # re-anchor after an overrun, no catch-up burst

            # Check if task is successful this timestep (10 consecutive timesteps in success state)
            # This is the same check that collect_demos uses
            if self.env.is_success()["task"]:
                if task_completion_hold_count > 0:
                    task_completion_hold_count -= 1
                else:
                    task_completion_hold_count = 10
            else:
                task_completion_hold_count = -1

            t += 1

            # Set done flag
            horizon_done = t >= max_horizon
            success = task_completion_hold_count == 0
            if success:
                print("TASK SUCCESSFUL!")
            done = env_done or success or horizon_done or obs is None

            if done:
                if not self.keep_if_unsuccessful and not success:
                    print("TASK UNSUCCESSFUL — DISCARDING EPISODE")
                    _finish_timing()
                    self.writer.abort_episode()
                    self.seed -= 1 # repeat same environment for next rollout since this one is discarded
                    return -1
                break

            # update for next iter
            state_dict = self.env.get_state()

        self.cached_state_dict = None
        _finish_timing()
        self.writer.log_episode(
            env=self.env,
            success=success,
        )
        self.viewer.close()
        return 0
    
    def get_lang_emb(self, env, device):
        ep_meta = env.env.get_ep_meta()
        ep_lang_str = ep_meta.get("lang", "dummy")
        print("TASK: ", ep_lang_str)
        lang_encoder = LangUtils.LangEncoder(
            device=device,
        )
        lang_emb = lang_encoder.get_lang_emb(ep_lang_str)
        if torch.is_tensor(lang_emb):
            lang_emb = lang_emb.cpu().numpy()
        lang_emb = np.squeeze(lang_emb)
        return lang_emb