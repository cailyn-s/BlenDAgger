import h5py
import json
import os
import time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from interfaces.data_writer import AbstractDataWriter

# Intervention magnitude: per-group L2 of clip(||human_action[group]||, 0, 1), as in the arbitrator
# (magnitude_scale=1) and the real-robot pipeline in the companion xArm repository.
_MOVING_FLOOR = 5e-4  # ||ha[:6]|| below this = not commanding (== the arbitrator's _MAGNITUDE_FLOOR)

def _intervention_magnitude(ha6):
    ha6 = np.asarray(ha6, dtype=np.float64)
    return float(np.linalg.norm([
        min(float(np.linalg.norm(ha6[0:2])), 1.0),
        min(float(np.linalg.norm(ha6[2:3])), 1.0),
        min(float(np.linalg.norm(ha6[3:6])), 1.0),
    ]))

# Per-DoF-group blend decomposition emitted by combined_blend* in arb["blend_terms"]; logged
# as "blend_<key>" per step ([xy,z,ori]), NaN on non-blend steps (same format as the real-robot writer).
_BLEND_TERM_KEYS = ("magnitude", "similarity", "confidence", "alpha_sim", "alpha_conf", "alpha_target", "alpha")


class DataWriter(AbstractDataWriter):
    def __init__(self, path, intervention_path, rollout_path, env_args, resume=False):
        self.path = path
        self.intervention_path = intervention_path
        self.rollout_path = rollout_path
        self.full_path = path.replace(".hdf5", "_full.hdf5")

        self.demo_id = 0
        self.segment_demo_id = 0
        self.lang_emb = None
        self.num_restarts = 0

        self.intervention_file = h5py.File(self.intervention_path, "a")
        if "data" not in self.intervention_file:
            self.intervention_grp = self.intervention_file.create_group("data")
            self.intervention_grp.attrs["env_args"] = json.dumps(env_args)
            self.intervention_grp.attrs["total"] = 0
        else:
            self.intervention_grp = self.intervention_file["data"]

        self.rollout_file = h5py.File(self.rollout_path, "a")
        if "data" not in self.rollout_file:
            self.rollout_grp = self.rollout_file.create_group("data")
            self.rollout_grp.attrs["env_args"] = json.dumps(env_args)
            self.rollout_grp.attrs["total"] = 0
        else:
            self.rollout_grp = self.rollout_file["data"]

        self.full_file = h5py.File(self.full_path, "a")
        if "data" not in self.full_file:
            self.full_grp = self.full_file.create_group("data")
            self.full_grp.attrs["env_args"] = json.dumps(env_args)
            self.full_grp.attrs["total"] = 0
        else:
            self.full_grp = self.full_file["data"]    
        
        self.episode_health = None  # loop-rate / input-lag stats from the collector

        self.meta_path = f"{self.path.replace('_dataset.hdf5', '')}_meta.json"
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.meta_data = json.load(f)
        else:
            self.meta_data = {}

        # Resuming an interrupted round: continue the counters instead of overwriting demo_0.
        # Aborted (_restartN) episodes don't advance demo_id, so they don't count here either.
        if resume:
            self.demo_id = sum(1 for v in self.meta_data.values()
                               if not v.get("summary", {}).get("aborted", False))
            existing = [int(n.split("_")[1]) for grp in (self.intervention_grp, self.rollout_grp)
                        for n in grp.keys() if n.startswith("demo_")]
            self.segment_demo_id = max(existing, default=-1) + 1
            print(f"[writer] resuming round at episode {self.demo_id}, "
                  f"segment id {self.segment_demo_id}")

    def set_language_embedding(self, lang_emb):
        self.lang_emb = lang_emb

    def set_episode_health(self, health):
        """Timing stats for the episode about to be closed; merged into the meta JSON."""
        self.episode_health = dict(health) if health else None

    def start_episode(self, create_grp=True):
        self.intervention_state_buffer = []
        self.intervention_action_buffer = []
        self.intervention_grasp_event_buffer = []
        self.intervention_post_grasp_buffer = []

        self.rollout_train_state_buffer = []
        self.rollout_train_action_buffer = []
        self.rollout_grasp_event_buffer = []
        self.rollout_post_grasp_buffer = []

        self.intervention_segments = []  
        self.rollout_segments = []

        self.full_state_buffer = []
        self.full_executed_action_buffer = []
        self.full_robot_action_buffer = []
        self.full_human_action_buffer = []
        self.full_control_mode_buffer = []
        self.full_train_mask_buffer = []
        self.full_intervened_buffer = []
        self.full_alpha_buffer = []
        self.full_blend_terms_buffer = []
        self.full_blend_scalars_buffer = []
        self.full_uncertainty_raw_buffer = []
        self.full_uncertainty_norm_buffer = []
        self.full_joint_pos_buffer = []
        self.full_joint_vel_buffer = []
        self.full_joint_acc_buffer = []
        self.full_eef_pos_buffer = []
        self.full_eef_quat_buffer = []
        self.full_gripper_pos_buffer = []

        self.intervention_magnitudes = []
        self.intervention_magnitudes_with_gripper = []
        self.intervention_magnitudes_moving = []
        self.grasp_event_buffer = []
        self.post_grasp_buffer = []

        self.clear_buffers()

    def log_step(
        self,
        executed_action,
        *,
        human_action=None,
        robot_action=None,
        control_mode="robot",
        gripper_active=False,
        intervening=False,
        training=False,
        alphas=None,
        metadata=None,
        blend_terms=None,
        uncertainty_raw=None,
        uncertainty_norm=None,
        sim_state=None,
        joint_pos=None,
        joint_vel=None,
        joint_acc=None,
        eef_pos=None,
        eef_quat=None,
        gripper_pos=None,
    ):
        self.ep_steps += 1

        if intervening:
            self.intervention_steps += 1
            if not self._prev_intervened:
                self.distinct_interventions += 1
            if human_action is not None:
                ha6 = np.asarray(human_action[0:6], dtype=np.float64)
                magnitude = _intervention_magnitude(ha6)  # per-group clip(||.||,0,1), matches arbitration magnitude
                self.intervention_magnitudes.append(magnitude)
                self.intervention_magnitudes_with_gripper.append(magnitude)  # key name is historical; value excludes the gripper
                if float(np.linalg.norm(ha6)) > _MOVING_FLOOR:  # moving-frame mean excludes hold-still/dropout frames
                    self.intervention_magnitudes_moving.append(magnitude)

        if sim_state is not None:
            self.full_train_mask_buffer.append(training)
            action = executed_action.detach().cpu().numpy() if hasattr(executed_action, "detach") else np.asarray(executed_action)
            
            ha = human_action.detach().cpu().numpy() if hasattr(human_action, "detach") else np.asarray(human_action) if human_action is not None else np.zeros_like(action)
            ra = robot_action.detach().cpu().numpy() if hasattr(robot_action, "detach") else np.asarray(robot_action) if robot_action is not None else np.zeros_like(action)
            alp = alphas if alphas is not None else [1.0]*6 # 1 if not intervening

            self.full_state_buffer.append(sim_state)
            if joint_pos is not None:
                self.full_joint_pos_buffer.append(joint_pos)
                self.full_joint_vel_buffer.append(joint_vel)
                self.full_joint_acc_buffer.append(joint_acc)
            if eef_pos is not None:
                self.full_eef_pos_buffer.append(eef_pos)
                self.full_eef_quat_buffer.append(eef_quat)
                self.full_gripper_pos_buffer.append(gripper_pos)
            self.full_executed_action_buffer.append(executed_action)
            self.full_alpha_buffer.append(alp)
            self.full_blend_terms_buffer.append(blend_terms)  # per-step blend decomposition (None on non-blend steps)
            _bm = metadata or {}   # per-step blend scalars: gate_env, onset, target_held
            self.full_blend_scalars_buffer.append({
                "gate_env": float(_bm.get("gate_env", np.nan)),
                "onset": bool(_bm.get("is_first_intervention_step", False)),
                "target_held": bool(_bm.get("target_held", False)),
            })
            self.full_uncertainty_raw_buffer.append(uncertainty_raw)   # per-step raw + normalized policy uncertainty
            self.full_uncertainty_norm_buffer.append(uncertainty_norm)

            # Save rollout and intervention data for any training steps
            if training:
                # Compute grasp logic
                gripper = executed_action[6]
                if self._prev_gripper is None:
                    grasp_event = False
                else:
                    grasp_event = abs(gripper - self._prev_gripper) > 0.01
                self._prev_gripper = gripper
                if grasp_event:
                    self._post_grasp_countdown = 3
                post_grasp = self._post_grasp_countdown > 0
                if self._post_grasp_countdown > 0:
                    self._post_grasp_countdown -= 1

                # Check for boundaries and save completed segments
                if intervening and not self._prev_intervened:
                    if len(self.current_roll_seg["states"]) > 0:
                        self.segments_to_save.append({"type": "rollout", "data": self.current_roll_seg})
                        self.current_roll_seg = {
                            "states": [], "actions": [], "human_actions": [], "robot_actions": [], "alphas": [],
                            "grasp_event_mask": [], "post_grasp_mask": []
                        }

                elif not intervening and self._prev_intervened:
                    if len(self.current_int_seg["states"]) > 0:
                        self.segments_to_save.append({"type": "intervention", "data": self.current_int_seg})
                        self.current_int_seg = {
                            "states": [], "actions": [], "human_actions": [], "robot_actions": [], "alphas": [],
                            "grasp_event_mask": [], "post_grasp_mask": []
                        }

                # Append to the original flat buffers and the current segment
                if intervening:
                    self.intervention_state_buffer.append(sim_state)
                    self.intervention_action_buffer.append(action)
                    self.intervention_grasp_event_buffer.append(grasp_event)
                    self.intervention_post_grasp_buffer.append(post_grasp)
                    
                    self.current_int_seg["states"].append(sim_state)
                    self.current_int_seg["actions"].append(action)
                    self.current_int_seg["human_actions"].append(ha)
                    self.current_int_seg["robot_actions"].append(ra)
                    self.current_int_seg["alphas"].append(alp)
                    self.current_int_seg["grasp_event_mask"].append(grasp_event)
                    self.current_int_seg["post_grasp_mask"].append(post_grasp)
                    
                if not intervening:
                    self.rollout_train_state_buffer.append(sim_state)
                    self.rollout_train_action_buffer.append(action)
                    self.rollout_grasp_event_buffer.append(grasp_event)
                    self.rollout_post_grasp_buffer.append(post_grasp)
                    
                    self.current_roll_seg["states"].append(sim_state)
                    self.current_roll_seg["actions"].append(action)
                    self.current_roll_seg["human_actions"].append(ha)
                    self.current_roll_seg["robot_actions"].append(ra)
                    self.current_roll_seg["alphas"].append(alp)
                    self.current_roll_seg["grasp_event_mask"].append(grasp_event)
                    self.current_roll_seg["post_grasp_mask"].append(post_grasp)
            
                self.grasp_event_buffer.append(grasp_event)
                self.post_grasp_buffer.append(post_grasp)
            else:
                self.grasp_event_buffer.append(False)
                self.post_grasp_buffer.append(False)
            
            # Save all data to full dataset
            self.full_robot_action_buffer.append(ra)
            self.full_human_action_buffer.append(ha)
            self.full_control_mode_buffer.append(control_mode)
            self.full_intervened_buffer.append(intervening)

        self._prev_intervened = intervening

    def log_episode(self, env, success):
        no_data = True

        # Flush the final active segment into the master list
        if len(self.current_int_seg["states"]) > 0:
            self.segments_to_save.append({"type": "intervention", "data": self.current_int_seg})
        if len(self.current_roll_seg["states"]) > 0:
            self.segments_to_save.append({"type": "rollout", "data": self.current_roll_seg})

        # Iterate chronologically and save to the relevant file
        for i, seg_info in enumerate(self.segments_to_save):
            seg_type = seg_info["type"]
            seg = seg_info["data"]
            
            states = np.asarray(seg["states"])
            if states.shape[0] == 0:
                continue

            demo_name = f"demo_{self.segment_demo_id}"
            actions = np.asarray(seg["actions"])
            grasp_mask = np.asarray(seg["grasp_event_mask"], dtype=np.bool_)
            post_grasp_mask = np.asarray(seg["post_grasp_mask"], dtype=np.bool_)

            # Route to the correct HDF5 group based on the segment type
            target_grp = self.intervention_grp if seg_type == "intervention" else self.rollout_grp

            if demo_name in target_grp:
                del target_grp[demo_name]

            demo_grp = target_grp.create_group(demo_name)
            demo_grp.create_dataset("states", data=states)
            demo_grp.create_dataset("actions", data=actions)
            demo_grp.create_dataset("human_actions", data=np.asarray(seg["human_actions"]))
            demo_grp.create_dataset("robot_actions", data=np.asarray(seg["robot_actions"]))
            demo_grp.create_dataset("alphas", data=np.asarray(seg["alphas"]))
            
            demo_grp.create_dataset("grasp_event_mask", data=grasp_mask)
            demo_grp.create_dataset("post_grasp_mask", data=post_grasp_mask)

            # Metadata
            xml_str = env.env.model.get_xml()
            demo_grp.attrs["model_file"] = xml_str
            ep_meta = env.env.get_ep_meta()
            demo_grp.attrs["ep_meta"] = json.dumps(ep_meta)
            demo_grp.attrs["num_samples"] = states.shape[0]
            demo_grp.attrs["source"] = seg_type
            demo_grp.attrs["original_demo_id"] = self.demo_id
            demo_grp.attrs["segment_index"] = i

            target_grp.attrs["total"] += states.shape[0]
            
            # Increment the shared counter only when a demo is written
            self.segment_demo_id += 1
            no_data = False

        if no_data:
            print("No training data logged for this episode!")

        if len(self.full_state_buffer) > 0:
            if no_data:  # no segment was written this episode, so reserve an id for the full demo
                demo_name = f"demo_{self.segment_demo_id}"
                self.segment_demo_id += 1

            if demo_name in self.full_grp:
                del self.full_grp[demo_name]

            demo_grp = self.full_grp.create_group(demo_name)

            demo_grp.create_dataset("states", data=np.asarray(self.full_state_buffer))
            demo_grp.create_dataset("actions", data=np.asarray(self.full_executed_action_buffer))
            demo_grp.create_dataset("robot_actions", data=np.asarray(self.full_robot_action_buffer))
            demo_grp.create_dataset("human_actions", data=np.asarray(self.full_human_action_buffer))
            demo_grp.create_dataset("train_mask", data=np.asarray(self.full_train_mask_buffer, dtype=np.bool_))
            demo_grp.create_dataset("intervention_mask", data=np.asarray(self.full_intervened_buffer, dtype=np.bool_))
            demo_grp.create_dataset("alphas", data=np.asarray(self.full_alpha_buffer))
            for _bk in _BLEND_TERM_KEYS:  # per-DoF blend decomposition, NaN on non-blend steps
                _arr = np.full((len(self.full_blend_terms_buffer), 3), np.nan, dtype=np.float32)
                for _i, _bt in enumerate(self.full_blend_terms_buffer):
                    if _bt is not None and _bk in _bt:
                        _arr[_i] = np.asarray(_bt[_bk], dtype=np.float32).reshape(-1)[:3]
                demo_grp.create_dataset(f"blend_{_bk}", data=_arr)
            demo_grp.create_dataset("blend_gate_env", data=np.array([b["gate_env"] for b in self.full_blend_scalars_buffer], dtype=np.float32))
            demo_grp.create_dataset("blend_onset_mask", data=np.array([b["onset"] for b in self.full_blend_scalars_buffer], dtype=np.bool_))
            demo_grp.create_dataset("blend_target_held", data=np.array([b["target_held"] for b in self.full_blend_scalars_buffer], dtype=np.bool_))
            for _un, _ubuf in (("uncertainty_raw", self.full_uncertainty_raw_buffer), ("uncertainty_norm", self.full_uncertainty_norm_buffer)):
                _uv = [np.asarray(u, dtype=np.float32).reshape(-1) for u in _ubuf if u is not None]
                if _uv:
                    _uD = max(v.shape[0] for v in _uv)
                    _uarr = np.full((len(_ubuf), _uD), np.nan, dtype=np.float32)
                    for _ui, _u in enumerate(_ubuf):
                        if _u is not None:
                            _uu = np.asarray(_u, dtype=np.float32).reshape(-1)
                            _uarr[_ui, :min(_uD, _uu.shape[0])] = _uu[:_uD]
                    demo_grp.create_dataset(_un, data=_uarr)
            demo_grp.create_dataset("grasp_event_mask", data=np.asarray(self.grasp_event_buffer, dtype=np.bool_))
            demo_grp.create_dataset("post_grasp_mask", data=np.asarray(self.post_grasp_buffer, dtype=np.bool_))
            if len(self.full_joint_pos_buffer) > 0:
                demo_grp.create_dataset("joint_pos", data=np.asarray(self.full_joint_pos_buffer))
                demo_grp.create_dataset("joint_vel", data=np.asarray(self.full_joint_vel_buffer))
                demo_grp.create_dataset("joint_acc", data=np.asarray(self.full_joint_acc_buffer))
            if len(self.full_eef_pos_buffer) > 0:  # observed eef pose + gripper, for trim-zeros
                demo_grp.create_dataset("eef_pos", data=np.asarray(self.full_eef_pos_buffer))
                demo_grp.create_dataset("eef_quat", data=np.asarray(self.full_eef_quat_buffer))
                demo_grp.create_dataset("gripper_pos", data=np.asarray(self.full_gripper_pos_buffer))

            control_mode_map = {"robot": 0, "human": 1, "blend": 2}
            encoded_modes = np.array(
                [control_mode_map[m] for m in self.full_control_mode_buffer],
                dtype=np.int32
            )
            demo_grp.create_dataset("control_mode", data=encoded_modes)

            demo_grp.attrs["task_success"] = bool(success)
            demo_grp.attrs["num_samples"] = len(self.full_state_buffer)
            if self.lang_emb is not None:
                demo_grp.attrs["lang_emb"] = self.lang_emb

            xml_str = env.env.model.get_xml()
            demo_grp.attrs["model_file"] = xml_str
            ep_meta = env.env.get_ep_meta()
            demo_grp.attrs["ep_meta"] = json.dumps(ep_meta)

            self.full_grp.attrs["total"] += len(self.full_state_buffer)

        # Flush per episode so an interrupted run keeps every finished episode readable.
        self.flush()

        self.end_episode(success)

    def end_episode(self, success, aborted=False):
        task_time_seconds = time.time() - self.ep_start_time

        episode_key = f"demo_{self.demo_id}"
        if aborted:
            episode_key += f"_restart{self.num_restarts}"

        summary = {
            "task_success": bool(success),
            "task_time_seconds": float(task_time_seconds),
            "aborted": aborted,
            "num_distinct_interventions": int(self.distinct_interventions),
            "num_intervention_steps": int(self.intervention_steps),
            "mean_intervention_magnitude": (
                float(np.mean(self.intervention_magnitudes))
                if len(self.intervention_magnitudes) > 0 else 0.0
            ),
            "mean_intervention_magnitude_moving": (
                float(np.mean(self.intervention_magnitudes_moving))
                if len(self.intervention_magnitudes_moving) > 0 else 0.0
            ),
            "total_intervention_magnitude": (
                float(np.sum(self.intervention_magnitudes))
                if len(self.intervention_magnitudes) > 0 else 0.0
            ),
            "mean_intervention_mangitude_with_gripper": (
                float(np.mean(self.intervention_magnitudes_with_gripper))
                if len(self.intervention_magnitudes_with_gripper) > 0 else 0.0
            ),
            "total_intervention_magnitude_with_gripper": (
                float(np.sum(self.intervention_magnitudes_with_gripper))
                if len(self.intervention_magnitudes_with_gripper) > 0 else 0.0
            ),
            "num_restarts": self.num_restarts
        }

        self.meta_data[episode_key] = {
            "summary": summary
        }
        if self.episode_health:
            self.meta_data[episode_key]["timing"] = self.episode_health
        self.episode_health = None

        temp_path = f"{self.meta_path}.tmp"
        with open(temp_path, "w") as f:
            json.dump(self.meta_data, f, indent=2)
        os.replace(temp_path, self.meta_path)

        if not aborted:
            self.demo_id += 1
            self.num_restarts = 0
        self.clear_buffers()

    def abort_episode(self):
        self.num_restarts += 1
        self.end_episode(success=False, aborted=True)

    def clear_buffers(self):
        self.intervention_state_buffer.clear()
        self.intervention_action_buffer.clear()
        self.intervention_grasp_event_buffer.clear()
        self.intervention_post_grasp_buffer.clear()

        self.rollout_train_state_buffer.clear()
        self.rollout_train_action_buffer.clear()
        self.rollout_grasp_event_buffer.clear()
        self.rollout_post_grasp_buffer.clear()

        self.full_state_buffer.clear()
        self.full_blend_terms_buffer.clear()
        self.full_blend_scalars_buffer.clear()
        self.full_uncertainty_raw_buffer.clear()
        self.full_uncertainty_norm_buffer.clear()
        self.full_executed_action_buffer.clear()
        self.full_robot_action_buffer.clear()
        self.full_human_action_buffer.clear()
        self.full_control_mode_buffer.clear()
        self.full_train_mask_buffer.clear()
        self.full_intervened_buffer.clear()
        self.full_alpha_buffer.clear()
        self.full_joint_pos_buffer.clear()
        self.full_joint_vel_buffer.clear()
        self.full_joint_acc_buffer.clear()
        self.full_eef_pos_buffer.clear()
        self.full_eef_quat_buffer.clear()
        self.full_gripper_pos_buffer.clear()
        self.grasp_event_buffer.clear()
        self.post_grasp_buffer.clear()

        self.segments_to_save = []
        self.current_int_seg = {
            "states": [], "actions": [], "human_actions": [], "robot_actions": [], "alphas": [],
            "grasp_event_mask": [], "post_grasp_mask": []
        }
        self.current_roll_seg = {
            "states": [], "actions": [], "human_actions": [], "robot_actions": [], "alphas": [],
            "grasp_event_mask": [], "post_grasp_mask": []
        }

        # reset variables
        self._prev_gripper = None
        self._post_grasp_countdown = 0
        self.ep_steps = 0
        self.intervention_steps = 0
        self.distinct_interventions = 0
        self.intervention_magnitudes.clear()
        self.intervention_magnitudes_with_gripper.clear()
        self.intervention_magnitudes_moving.clear()
        self._prev_intervened = False
        self.ep_start_time = time.time()

    def finalize(self):
        def _write_mask(f, grp):
            mask_grp = f.require_group("mask")
            if "train" in mask_grp:   # reopened file: the old mask is stale, not a conflict
                del mask_grp["train"]
            mask_grp.create_dataset("train", data=np.array(list(grp.keys()), dtype="S"))

        _write_mask(self.intervention_file, self.intervention_grp)
        _write_mask(self.rollout_file, self.rollout_grp)
        if self.full_file is not None and "data" in self.full_file:
            _write_mask(self.full_file, self.full_grp)

    def flush(self):
        """Make everything written so far readable on disk, without closing."""
        for f in (self.intervention_file, self.rollout_file, self.full_file):
            if f is not None:
                try:
                    f.flush()
                except Exception as e:      # never let a flush failure kill a collection
                    print(f"[writer] flush failed: {type(e).__name__}: {e}")

    def close(self):
        self.finalize()

        if self.intervention_file is not None:
            self.intervention_file.close()
            self.intervention_file = None

        if self.rollout_file is not None:
            self.rollout_file.close()
            self.rollout_file = None

        if self.full_file is not None:
            self.full_file.close()
            self.full_file = None

    # ------------------------------------------------------------------
    # AbstractDataWriter bridge methods
    # ------------------------------------------------------------------

    def open_episode(self, episode_id: int) -> None:
        self.start_episode(create_grp=True)

    def write_step(self, obs: np.ndarray, action: np.ndarray, metadata: dict) -> None:
        self.log_step(
            executed_action=action,
            human_action=metadata.get("human_action"),
            robot_action=metadata.get("robot_action"),
            control_mode=metadata.get("control_mode", "robot"),
            gripper_active=metadata.get("gripper_active", False),
            intervening=metadata.get("intervening", False),
            training=metadata.get("training", False),
            alphas=metadata.get("alphas"),
            metadata=metadata,
            sim_state=metadata.get("sim_state"),
        )

    def close_episode(self, success: bool, score: int = 0) -> None:
        if score < 0:
            self.abort_episode()
        else:
            self.log_episode(env=self._env_ref, success=success)

    def set_env_ref(self, env) -> None:
        """Call once after construction so close_episode can pass env to log_episode."""
        self._env_ref = env