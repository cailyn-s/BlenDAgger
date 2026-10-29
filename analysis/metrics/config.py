"""Composition for the per-consumer config split.

conf/<task>.yaml lists `includes:` naming conf/<task>/{run,compute,plot,suite}.yaml,
so the plotting side can be packaged with only run.yaml + plot.yaml. This is not
hydra's `defaults:` -- these configs are read with a plain OmegaConf.load, so the
merge happens here. Anything written directly in the including file wins over the
parts, which keeps a one-line per-task override working.
"""
import os
import sys

from omegaconf import OmegaConf

# this file is <repo>/analysis/metrics/config.py, so conf/ is three levels up
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONF = os.path.join(_REPO, "conf")
sys.path.insert(0, _REPO)
from repo_paths import load_paths_db  # noqa: E402


def load_task_config(cfg_path: str, conf_dir: str | None = None):
    """Load a task config, merging any `includes:` it lists (later entries win).

    `paths_db` is merged in underneath, so a config can write `${paths_db.real_data_dir}`.
    """
    conf_dir = conf_dir or CONF
    cfg = OmegaConf.load(cfg_path)
    includes = cfg.pop("includes", None)
    base = OmegaConf.create({"paths_db": load_paths_db()})
    if not includes:
        return OmegaConf.merge(base, cfg)
    merged = base
    for rel in list(includes):
        part = os.path.join(conf_dir, f"{rel}.yaml")
        if not os.path.exists(part):
            raise FileNotFoundError(f"config include not found: {part} (from {cfg_path})")
        merged = OmegaConf.merge(merged, OmegaConf.load(part))
    return OmegaConf.merge(merged, cfg)
