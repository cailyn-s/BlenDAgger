"""Repo-relative path helpers shared by the scripts that are not launched through hydra."""
import os
import re

import numpy as np
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(REPO, "conf")
LOCAL_PATHS = os.path.join(CONF, "local", "paths.yaml")


def resolve(path):
    """`path` made absolute against the repo root (absolute paths pass through)."""
    return path if path is None or os.path.isabs(path) else os.path.normpath(os.path.join(REPO, path))


def load_paths_db(resolved=True):
    """conf/paths_db.yaml merged with conf/local/paths.yaml, the same composition hydra does."""
    cfg = OmegaConf.load(os.path.join(CONF, "paths_db.yaml"))
    cfg.pop("defaults", None)
    if os.path.exists(LOCAL_PATHS):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(LOCAL_PATHS))
    if not resolved:
        return cfg.paths_db
    paths = OmegaConf.to_container(cfg, resolve=True)["paths_db"]
    return OmegaConf.create(_absolute(paths))


def _absolute(node):
    if isinstance(node, dict):
        return {k: _absolute(v) for k, v in node.items()}
    return resolve(node) if isinstance(node, str) else node


# A saved env state embeds the absolute asset paths of the machine that saved it.
_ASSET_ROOTS = {
    "robocasa": re.compile(r"[^\"'\s]*?robocasa/models/assets/"),
    "robosuite": re.compile(r"[^\"'\s]*?robosuite/models/assets/"),
}


def reroot_assets(text):
    """Point every robocasa / robosuite asset path in `text` at this machine's installs."""
    import robocasa.models
    import robosuite.models
    roots = {"robocasa": robocasa.models.assets_root, "robosuite": robosuite.models.assets_root}
    for pkg, pattern in _ASSET_ROOTS.items():
        text = pattern.sub(roots[pkg].rstrip("/") + "/", text)
    return text


def load_fixed_env(path):
    """A fixed_envs/*.npy state dict, with its asset paths valid on this machine."""
    state = np.load(resolve(path), allow_pickle=True).item()
    for key in ("model", "ep_meta"):
        if isinstance(state.get(key), str):
            state[key] = reroot_assets(state[key])
    return state
