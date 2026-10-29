"""Browse candidate fixed env scenes with teleop, and save the one you like.

Each candidate is a seed. The env's rng is reseeded before every reset, so a seed always
produces the same layout/style -- which is what makes the printed seed worth recording.
Drive the robot to check the task is actually doable, then save or skip.

  python scripts/scenes/browse_fixed_envs.py --task PrepareCoffee --name coffee

Keys (focus the OpenCV window):
  n  next seed          k  keep/save this scene   r  re-randomize object position
  p  previous seed      ESC  quit

The teleop devices install a GLOBAL keyboard listener, so their keys fire even when this
window has focus. Reserved by them -- do not rebind onto these: b (toggle arm/base mode),
s (switch arm), = (switch robot), q, space, shift.
"""
import argparse
import json
import os
import time

import sys

import cv2
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.env_utils import create_env_from_metadata
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robocasa.utils.dataset_registry import get_ds_path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)
from collection.randomize_pos import randomize_for_task  # noqa: E402
from collection.multiview import MultiViewRenderer  # noqa: E402

# Tasks this tool can browse; each one's env metadata comes from its robocasa human dataset.
TASKS = ["PrepareCoffee", "MicrowaveThawing", "PnPCounterToCab"]

HUD = "n next | p prev | r re-randomize | k keep/save | ESC quit"


def compose(env, cameras, height, width, seed, layout, style, saved):
    """Side-by-side camera views with a seed/layout/style banner."""
    imgs = []
    for cam in cameras:
        img = env.env.sim.render(camera_name=cam, height=height, width=width)
        img = cv2.flip(img, 0)  # MuJoCo renders upside down
        imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    frame = np.concatenate(imgs, axis=1)

    banner = np.zeros((70, frame.shape[1], 3), dtype=np.uint8)
    label = f"seed {seed}   layout {layout}   style {style}"
    if saved:
        label += "   [SAVED]"
    cv2.putText(banner, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(banner, HUD, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return np.concatenate([banner, frame], axis=0)


def load_seed(env, seed):
    """Reseed then reset, so this seed's layout/style are reproducible."""
    env.env.rng = np.random.default_rng(seed)
    obs = env.reset()
    return obs, int(env.env.layout_id), int(env.env.style_id)


def pristine_state(env, seed):
    """The scene exactly as reset() leaves it, before any teleop or randomization.

    This is what gets saved. Driving the robot to test a scene moves the objects -- doing
    the task parks the mug under the dispenser -- and saving the live state would bake
    that in, destroying the task's starting condition.
    """
    env.env.rng = np.random.default_rng(seed)
    env.reset()
    return env.get_state()


def save_scene(state, out_npy, seed, layout, style, dataset, task):
    np.save(out_npy, state)
    meta = {
        "seed": seed, "layout_id": layout, "style_id": style,
        "task": task, "dataset": dataset,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.splitext(out_npy)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSAVED {out_npy}  (seed={seed}, layout={layout}, style={style})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--name", help="output name, e.g. 'coffee' -> fixed_envs/fixed_env_state_coffee.npy")
    ap.add_argument("--dataset_path", help="override the dataset used for env metadata")
    ap.add_argument("--start_seed", type=int, default=0)
    ap.add_argument("--device", default="spacemouse", choices=["spacemouse", "keyboard"])
    ap.add_argument("--res", type=int, default=700, help="per-camera render size")
    args = ap.parse_args()

    name = args.name or args.task.lower()
    dataset = args.dataset_path or get_ds_path(args.task, ds_type="human_im")
    # anchor on the repo so the save works regardless of cwd
    repo = REPO
    out_npy = os.path.join(repo, "fixed_envs", f"fixed_env_state_{name}.npy")

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": [], "rgb": ["robot0_agentview_right_image", "robot0_eye_in_hand_image"], "depth": []}}
    )
    meta = get_env_metadata_from_dataset(dataset)
    meta["env_kwargs"]["layout_ids"] = [-1]
    meta["env_kwargs"]["style_ids"] = [-1]
    env = create_env_from_metadata(env_meta=meta, render=False, render_offscreen=True, use_image_obs=True)

    # teleop wraps the raw robosuite env; import here so --help works without hardware
    from main import make_teleop
    teleop, _ = make_teleop(env, args.device)
    cameras = MultiViewRenderer(env).cameras

    print(f"\nBrowsing {args.task} from seed {args.start_seed}. Will save to {out_npy}")
    print(HUD + "\n")

    seed = args.start_seed
    obs, layout, style = load_seed(env, seed)
    teleop.reset()
    prev_action, saved, quit_now = None, False, False

    loop_dt = 1.0 / 20.0  # match collector.py's 20 Hz control loop so teleop feels the same

    while not quit_now:
        tick = time.time()
        cv2.imshow("Scene Browser", compose(env, cameras, args.res, args.res, seed, layout, style, saved))
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("n"), ord("p")):
            seed += 1 if key == ord("n") else -1
            obs, layout, style = load_seed(env, seed)
            teleop.reset()
            prev_action, saved = None, False
            print(f"seed {seed}  layout {layout}  style {style}")
        elif key == ord("r"):
            # same policy collector.py applies each episode, so the preview is honest
            try:
                randomize_for_task(env, args.task, seed=np.random.randint(1 << 30))
            except Exception as e:
                print(f"  randomize failed ({type(e).__name__}: {e}) -- scene unchanged")
        elif key == ord("k"):
            # save the scene as reset() leaves it, not whatever test-driving left behind
            save_scene(pristine_state(env, seed), out_npy, seed, layout, style, dataset, args.task)
            obs = env.reset_to(np.load(out_npy, allow_pickle=True).item())
            teleop.reset()
            prev_action = None
            saved = True
        elif key == 27:  # ESC
            quit_now = True
        else:
            action, _, _ = teleop.get_action(obs, None, prev_action)
            if action is not None:
                obs, _, _, _ = env.step(action)
                prev_action = action

        time.sleep(max(0.0, loop_dt - (time.time() - tick)))

    cv2.destroyAllWindows()
    env.close()
    if os.path.exists(out_npy):
        print(f"\nDone. conf yaml already expects fixed_envs/fixed_env_state_{name}.npy")
    else:
        print("\nDone. Nothing saved.")


if __name__ == "__main__":
    main()
