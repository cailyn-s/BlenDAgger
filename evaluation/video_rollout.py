# Adapted from robomimic.utils.train_utils.run_rollout, but allows a fixed env state.
import numpy as np
import torch
import traceback
import random
from copy import deepcopy
from collections import deque
import robomimic.utils.lang_utils as LangUtils

from robomimic.utils.train_utils import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from collection.randomize_pos import randomize_obj_pos, randomize_for_task

def run_rollout(
        policy, 
        env, 
        horizon,
        task,
        state_dict=None,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
        seed=None,
        use_sequence=False,
        seq_length=1,
        device="cuda:0"
    ):
    """Runs a rollout in an environment with the current network parameters."""
    
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    policy.policy.set_eval()

    # Initialize buffer if using sequence model (e.g. transformer or diffusion)
    if use_sequence:
        obs_buffer = deque(maxlen=seq_length)
    else:
        obs_buffer = None

    # Seed global RNGs
    np.random.seed(seed)
    random.seed(seed)
    # Seed Robosuite/Robocasa's internal RandomState specifically
    if hasattr(env, "env") and hasattr(env.env, "np_random"):
        env.env.np_random = np.random.RandomState(seed)

    ob_dict = env.reset()
    if state_dict is not None:
        ob_dict = env.reset_to(state_dict)
        # Same per-task randomization the collector applies, so eval matches the training distribution.
        randomize_for_task(env, task, seed=seed)

    # Get language embedding and populate initial buffer
    lang_emb = get_lang_emb(env, device=device)
    if lang_emb is not None:
        ob_dict["lang_emb"] = lang_emb
        
    if obs_buffer is not None:
        for _ in range(obs_buffer.maxlen):
            obs_buffer.append(ob_dict)

    goal_dict = None
    if use_goals:
        goal_dict = env.get_goal()

    # Pass lang_emb to start_episode
    try:
        policy.start_episode(lang=lang_emb)
    except TypeError:
        policy.start_episode() # Fallback for standard robomimic RolloutPolicy

    results = {}
    video_count = 0  
    rews = []
    success = None 
    end_step = None
    video_frames = []

    # Initialize success counter
    task_completion_hold_count = -1
    subtask_ever = {}
    subtask_final = {}   # keys ending in _at_end: value at the final timestep, not latched
    
    try:
        for step_i in range(horizon):
            
            # Stack buffer for transformer, otherwise use raw dict
            if obs_buffer is not None:
                policy_ob = {k: np.stack([o[k] for o in obs_buffer], axis=0) for k in ob_dict.keys()}
            else:
                policy_ob = ob_dict
                
            # get action from policy
            ac = policy(ob=policy_ob, goal=goal_dict)
            ac = np.clip(ac, -1.0, 1.0) # ensure action is in valid range, get rid of floating point errors

            # play action
            ob_dict, r, done, _ = env.step(ac)

            # Update the buffer for the next timestep
            if lang_emb is not None:
                ob_dict["lang_emb"] = lang_emb
            if obs_buffer is not None:
                obs_buffer.append(ob_dict)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            rews.append(r)

            # Check if task is successful this timestep (10 consecutive timesteps in success state)
            # This is the same check that collect_demos uses
            success_info = env.is_success()
            # Subtasks are latched (achieved at any step counts); "task" keeps its 10-step hold rule below.
            for _k, _v in success_info.items():
                if _k == "task":
                    continue
                if _k.endswith("_at_end"):
                    # last-timestep semantics: report the state the episode ended in
                    subtask_final[_k] = bool(_v)
                else:
                    subtask_ever[_k] = subtask_ever.get(_k, False) or bool(_v)
            if success_info["task"]:
                if task_completion_hold_count > 0:
                    task_completion_hold_count -= 1
                else:
                    task_completion_hold_count = 10
            else:
                task_completion_hold_count = -1
            success = task_completion_hold_count == 0

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    frame = env.render(mode="rgb_array", height=512, width=512)
                    video_frames.append(frame)
                video_count += 1

            # break if done
            if done or (terminate_on_success and success):
                end_step = step_i
                break

    except Exception as e:
        print("\n--- ROLLOUT EXCEPTION ---")
        traceback.print_exc()
        print("------------------------------\n")

    # If rollout crashed on step 0, return zero success
    if success is None:
        print("WARNING: Rollout crashed before step 1 completed. Returning 0 success.")
        return {"Return": 0.0, "Horizon": 0, "Success_Rate": 0.0}

    if video_writer is not None:
        for frame in video_frames:
            video_writer.append_data(frame)

    end_step = end_step or step_i
    total_reward = np.sum(rews[:end_step + 1])
    
    results["Return"] = total_reward
    results["Horizon"] = end_step + 1
    results["Success_Rate"] = float(success)

    # log additional success metrics (latched over the episode, see above)
    for k, v in subtask_ever.items():
        results["{}_Success_Rate".format(k)] = float(v)
    # ...and the *_at_end ones, which report the final timestep instead
    for k, v in subtask_final.items():
        results["{}_Success_Rate".format(k)] = float(v)

    return results


_LANG_ENCODER = None      # load CLIP once, not per rollout
_LANG_EMB_CACHE = {}      # lang string -> embedding

def get_lang_emb(env, device):
    """Same as what is used in Collector script."""
    ep_meta = env.env.get_ep_meta()
    ep_lang_str = ep_meta.get("lang", "dummy")
    if ep_lang_str in _LANG_EMB_CACHE:
        return _LANG_EMB_CACHE[ep_lang_str]

    global _LANG_ENCODER
    if _LANG_ENCODER is None:
        _LANG_ENCODER = LangUtils.LangEncoder(device=device)
    lang_emb = _LANG_ENCODER.get_lang_emb(ep_lang_str)

    if torch.is_tensor(lang_emb):
        lang_emb = lang_emb.cpu().numpy()
    lang_emb = np.squeeze(lang_emb)
    _LANG_EMB_CACHE[ep_lang_str] = lang_emb
    return lang_emb