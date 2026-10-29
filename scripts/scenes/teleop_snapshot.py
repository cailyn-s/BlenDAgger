"""Teleop a collection scene and save high-resolution snapshots of it.

Renders figure snapshots only; not needed to reproduce any results.

The scene is built the way phase_1_collect_data + Collector.rollout build it (env kwargs,
fixed-env state, per-episode seed, randomize_for_task), so --round/--episode reproduce the
scene that episode was collected in. Each capture also saves the sim state, which --state
re-renders later with other cameras or sizes.

  python scripts/scenes/teleop_snapshot.py --config-name sim/thawing/blendagger --round 1 --episode 0
  python scripts/scenes/teleop_snapshot.py --config-name sim/thawing/blendagger --state snapshots/<...>_state.npy --cameras robot0_frontview

Keys (focus the OpenCV window):  c capture   ] next scene   [ previous scene   ESC quit
A SpaceMouse both-button hold (keyboard: q) resets the current scene, as in collection.
"""
import argparse
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)

import cv2
import numpy as np

from collection.randomize_pos import randomize_for_task
from repo_paths import load_fixed_env
from collection.multiview import PANE_H, PANE_W, MultiViewRenderer

WINDOW = "Teleop Snapshot"
LOOP_PERIOD = (1.0 / 20.0) / 0.4  # Collector's loop: 20 Hz control at 0.4x playback
# The three views the policy is trained on (conf/robomimic/finetune.json).
# The viewer only shows two of them, so agentview_left is captured but never displayed.
POLICY_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def load_config(name):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.join(REPO, "conf"), version_base=None):
        return compose(config_name=name)


def episode_seed(round_, episode):
    # Collector starts at 1000*(round+1) + start_episode and increments before each rollout.
    return 1000 * (round_ + 1) + episode + 1


def make_env(cfg, round_):
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.utils.env_utils import create_env_from_metadata
    from robomimic.utils.file_utils import get_env_metadata_from_dataset

    # Collection gets this from the policy checkpoint; image obs fail without it.
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": [], "rgb": ["robot0_agentview_right_image", "robot0_eye_in_hand_image"], "depth": []}}
    )
    base = 1000 * (round_ + 1)
    random.seed(base)
    meta = get_env_metadata_from_dataset(cfg.paths.pretrained_dataset)
    meta["env_kwargs"].update(layout_ids=[-1], seed=base, style_ids=[-1], control_freq=20)
    return create_env_from_metadata(meta, render=False, use_image_obs=True)


def reset_scene(env, task, fixed_state, seed):
    np.random.seed(seed)
    random.seed(seed)
    env.env.np_random = np.random.RandomState(seed)
    if fixed_state is None:
        env.reset()
        return
    env.reset_to(fixed_state)
    randomize_for_task(env, task, seed=seed)


def render(env, camera, width, height):
    img = env.env.sim.render(camera_name=camera, width=width, height=height)
    return cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR)  # MuJoCo renders upside down


def save_snapshot(env, out_dir, cameras, width, height, info):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{info.get('tag', 'snapshot')}_{time.strftime('%Y%m%d-%H%M%S')}")
    for cam in cameras:
        cv2.imwrite(f"{stem}_{cam}.png", render(env, cam, width, height))
    np.save(f"{stem}_state.npy", env.get_state())
    with open(f"{stem}.json", "w") as f:
        json.dump({**info, "cameras": cameras, "width": width, "height": height}, f, indent=2)
    print(f"[snapshot] saved {len(cameras)} image(s) at {width}x{height}: {stem}_*.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-name", required=True, help="collection config in conf/, e.g. sim/thawing/blendagger")
    ap.add_argument("--round", type=int, default=1, help="collection round the scene seed comes from")
    ap.add_argument("--episode", type=int, default=0, help="0-indexed episode within the round")
    ap.add_argument("--device", choices=["spacemouse", "keyboard"], help="default: the config's input_device")
    ap.add_argument("--cameras", nargs="+", help="cameras to capture (default: the three policy views)")
    # Same 16:9 aspect as the collection pane. MuJoCo fixes vertical FOV, so this keeps the framing.
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--out_dir", help="default: snapshots/<task>/")
    ap.add_argument("--state", help="re-render a saved *_state.npy instead of launching teleop")
    args = ap.parse_args()

    cfg = load_config(args.config_name)
    task = cfg.env.name
    out_dir = args.out_dir or os.path.join(REPO, "snapshots", task)
    env = make_env(cfg, args.round)
    sim = env.env.sim
    view_cams = MultiViewRenderer(env).cameras  # on-screen panes, not what gets captured
    cameras = args.cameras or POLICY_CAMERAS
    print("available cameras:", [sim.model.camera_id2name(i) for i in range(sim.model.ncam)])

    if args.state:
        env.reset_to(load_fixed_env(args.state))
        tag = os.path.basename(args.state)[:-len("_state.npy")] + "_rerender"
        save_snapshot(env, out_dir, cameras, args.width, args.height,
                      {"tag": tag, "config": args.config_name, "task": task, "source_state": args.state})
        env.close()
        return

    fixed_state = None
    if cfg.env.fixed_env:
        fixed_state = load_fixed_env(cfg.env.fixed_env)

    from main import make_teleop  # needs the device attached, so only on the teleop path
    teleop, _ = make_teleop(env, args.device or cfg.collection.input_device)

    episode = args.episode
    reset_scene(env, task, fixed_state, episode_seed(args.round, episode))
    teleop.reset()
    step, prev_action = 0, None
    print(f"\n{task}: round {args.round} episode {episode} (seed {episode_seed(args.round, episode)})")
    print("c capture | ] next scene | [ previous scene | ESC quit\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, PANE_W * len(view_cams), PANE_H)
    try:
        while True:
            tick = time.time()
            frame = np.concatenate([render(env, cam, PANE_W, PANE_H) for cam in view_cams], axis=1)
            label = f"round {args.round}  ep {episode}  seed {episode_seed(args.round, episode)}  step {step}"
            cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            if key == ord("c"):
                save_snapshot(env, out_dir, cameras, args.width, args.height, {
                    "tag": f"{args.config_name}_r{args.round}_ep{episode}_step{step:04d}",
                    "config": args.config_name, "task": task, "round": args.round, "episode": episode,
                    "seed": episode_seed(args.round, episode), "step": step,
                })
                continue

            reset = False
            if key in (ord("]"), ord("[")):
                episode = max(0, episode + (1 if key == ord("]") else -1))
                reset = True
            else:
                action, _, _ = teleop.get_action(None, None, prev_action)
                if action is None:  # device reset gesture
                    reset = True
                else:
                    env.step(action)
                    prev_action = action
                    step += 1

            if reset:
                reset_scene(env, task, fixed_state, episode_seed(args.round, episode))
                teleop.reset()
                step, prev_action = 0, None
                print(f"scene: round {args.round} episode {episode} (seed {episode_seed(args.round, episode)})")

            time.sleep(max(0.0, LOOP_PERIOD - (time.time() - tick)))
    finally:
        cv2.destroyAllWindows()
        try:
            teleop.device.close()
        except Exception as e:
            print("Error closing teleop device:", e)
        env.close()


if __name__ == "__main__":
    main()
