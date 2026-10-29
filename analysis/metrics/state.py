"""Analysis-run state that both computation and rendering have to agree on.

ANGLE_SPACE is set once per run from the obs configuration and then read on
both sides -- by the OOD/trajectory computation to pick a metric space, and by
the figures to pick axis units. It lives here so there is exactly one copy;
read it as `state.ANGLE_SPACE` rather than importing the value, which would
snapshot it at import time and miss `set_angle_space`.
"""

ANGLE_SPACE = False        # True when the obs state is PURELY joint angles


def set_angle_space(flag: bool) -> None:
    """Global switch for joint-angle obs -- see the `angle_keys` config key."""
    global ANGLE_SPACE
    ANGLE_SPACE = bool(flag)


def angle_space_for(obs_keys, angle_keys) -> bool:
    """Whether the obs state is PURELY joint angles.

    True only when angle_keys is non-empty and every obs key is one of them --
    a mixed state (arm_pos + arm_quat + arm_qpos) is not angle space, because
    the components are not commensurate. This is the single definition of the
    rule; run_analysis derives it here, and downstream plotting reads the
    answer back out of distributions_meta.json.
    """
    angle_keys = set(angle_keys or ())
    return bool(angle_keys) and set(obs_keys or ()) <= angle_keys

