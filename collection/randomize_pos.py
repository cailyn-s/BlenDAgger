import numpy as np
import json
import copy

def randomize_obj_pos(env, xy_range=0.1, z_offset=0.0, yaw_offset=np.pi, seed=None, change_base_pos=False):
    sim = unwrap_env(env)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    # Random XY offset
    dx = rng.uniform(-xy_range, xy_range)
    dy = rng.uniform(-xy_range, xy_range)
    dz = rng.uniform(-z_offset, z_offset)
    if not change_base_pos:
        # Change the position of the object to be picked up
        d_theta = rng.uniform(-yaw_offset, yaw_offset)
        change_object_position(sim.sim, "obj_main", dx=dx, dy=dy, dz=dz, d_theta=d_theta)
    else:
        # Change the position of the robot itself instead of the position of the articulated object
        yaw_offset = np.pi/12
        d_theta = rng.uniform(-yaw_offset, yaw_offset)
        change_robot_position(sim.sim, "mobilebase0_base", dx=dx, dy=dy, dz=dz, d_theta=d_theta)

def unwrap_env(env):
    level = 0
    current = env

    while True:

        if hasattr(current, "env"):
            current = current.env
            level += 1
        else:
            break


    return current

def change_object_position(sim, body_name, dx=0.0, dy=0.0, dz=0.0, d_theta=0.0):
    body_id = sim.model.body_name2id(body_name)

    # Get the joint attached to this body
    joint_id = sim.model.body_jntadr[body_id]

    if joint_id == -1:
        raise ValueError("Body has no joint — likely fixed in XML.")

    joint_type = sim.model.jnt_type[joint_id]

    if joint_type != 0:
        raise ValueError("Joint is not free — cannot reposition via qpos.")

    # Free joint occupies 7 qpos entries (x,y,z, quat)
    qpos_addr = sim.model.jnt_qposadr[joint_id]

    # Current pose
    current_pos = sim.data.qpos[qpos_addr:qpos_addr+3].copy()

    new_pos = current_pos.copy()
    new_pos[0] += dx
    new_pos[1] += dy
    new_pos[2] += dz
    sim.data.qpos[qpos_addr : qpos_addr + 3] = new_pos

    # ---- Orientation (yaw only) ----
    if d_theta != 0.0:
        # current quaternion
        quat_addr = qpos_addr + 3
        current_quat = sim.data.qpos[quat_addr : quat_addr + 4].copy()

        # Extract current yaw from quaternion
        # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        w, x, y, z = current_quat
        current_yaw = np.arctan2(
            2 * (w*z + x*y),
            1 - 2 * (y*y + z*z)
        )

        new_yaw = current_yaw + d_theta

        # Construct new pure-yaw quaternion
        qw = np.cos(new_yaw / 2)
        qx = 0.0
        qy = 0.0
        qz = np.sin(new_yaw / 2)

        sim.data.qpos[quat_addr : quat_addr + 4] = np.array([qw, qx, qy, qz])

    sim.forward()

    print(f"Moved {body_name} to {new_pos}, yaw += {d_theta} rad")

def change_robot_position(sim, body_name, dx=0.0, dy=0.0, dz=0.0, d_theta=0.0):
    body_id = sim.model.body_name2id(body_name)

    # Get the joint attached to this body
    joint_id = sim.model.body_jntadr[body_id]

    if joint_id == -1:
        raise ValueError("Body has no joint — likely fixed in XML.")
    
    joint_x = "mobilebase0_joint_mobile_forward"
    joint_y = "mobilebase0_joint_mobile_side"
    joint_yaw = "mobilebase0_joint_mobile_yaw"

    # Compute new pose
    new_x = sim.data.get_joint_qpos(joint_x) + dx
    new_y = sim.data.get_joint_qpos(joint_y) + dy
    new_yaw = sim.data.get_joint_qpos(joint_yaw) + d_theta

    sim.data.set_joint_qpos(joint_x, new_x)
    sim.data.set_joint_qpos(joint_y, new_y)
    sim.data.set_joint_qpos(joint_yaw, new_yaw)

    # Push the changes to the simulation before taking a step
    sim.forward()
    print(f"Moved {body_name} to {new_x}, {new_y}, yaw += {d_theta} rad")

# Sign of the mobile-base "forward" joint that moves the robot AWAY from the task fixture.
_BASE_BACK_SIGN = -1.0

# Per-task randomization for fixed-env collection.
#
# `bodies` move together by ONE shared delta, so a plate and the food resting on it stay
# aligned. Yaw is left at 0 for graspable objects: rotating a mug or plate much makes it
# ungraspable for a policy that never saw that pose.
#
# The base jitter is biased backward (away from the fixture) so the robot is less likely
# to be jammed against a cabinet, but the ranges are deliberately small -- these policies
# are not trained for mobile manipulation, so the object and target fixture have to stay
# inside the arm's demonstrated reach.
TASK_RANDOMIZATION = {
    "PrepareCoffee": dict(
        bodies=("obj_main",),                       # mug; coffee machine is a welded fixture
        # mug stays flat on the cabinet shelf within this range
        obj_xy=0.10, obj_yaw=0.0,
        base_side=0.04, base_back=(0.0, 0.05), base_yaw=np.pi / 18,
    ),
    "ArrangeVegetables": dict(
        # Not a paper task; vegetables and cutting board jitter as independent groups, ranges (x, y).
        groups=[(("vegetable1_main", "vegetable2_main"), (0.12, 0.04)),
                (("cutting_board_main",), 0.08)],
        obj_yaw=0.0,
        base_side=0.04, base_back=(0.0, 0.03), base_yaw=np.pi / 18,
    ),
    "MicrowaveThawing": dict(
        bodies=("obj_container_main", "obj_main"),  # plate + the food on it, moved as one
        obj_xy=0.05, obj_yaw=0.0,
        base_side=0.04, base_back=(0.0, 0.03), base_yaw=np.pi / 18,
    ),
}


def _body_geoms(sim, bodies):
    ids = {sim.model.body_name2id(b) for b in bodies}
    return {g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] in ids}


def _intersecting(sim, geoms, pen_thresh=-0.002):
    """True if a moved body is jammed into other geometry.

    Resting contact is fine (penetration ~0); a real intersection shows up as a deeply
    negative contact distance, and MuJoCo resolves those by launching the object.
    Contacts among the moved bodies themselves (food on its plate) are ignored.
    """
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if (c.geom1 in geoms) != (c.geom2 in geoms) and c.dist < pen_thresh:
            return True
    return False


def _free_joint_qpos(sim, body):
    j = sim.model.body_jntadr[sim.model.body_name2id(body)]
    if j == -1 or sim.model.jnt_type[j] != 0:
        return None
    a = sim.model.jnt_qposadr[j]
    return a, sim.data.qpos[a:a + 7].copy()


def randomize_for_task(env, task, seed=None, max_tries=10):
    """Randomize a fixed-env scene according to the task's policy.

    Tasks without an entry fall back to randomize_obj_pos: jitter the base for articulated
    tasks, the main object otherwise.
    """
    cfg = TASK_RANDOMIZATION.get(task)
    if cfg is None:
        change_base_pos = task in ("OpenSingleDoor", "OpenDrawer") or "Turn" in task
        randomize_obj_pos(env, seed=seed, change_base_pos=change_base_pos)
        return

    rng = np.random.RandomState(seed) if seed is not None else np.random
    sim = unwrap_env(env).sim
    groups = cfg.get("groups") or [(cfg["bodies"], cfg["obj_xy"])]
    groups = [([b for b in g if _free_joint_qpos(sim, b) is not None], r) for g, r in groups]
    groups = [(g, r) for g, r in groups if g]
    bodies = [b for g, _ in groups for b in g]
    saved = {b: _free_joint_qpos(sim, b) for b in bodies}
    geoms = _body_geoms(sim, bodies)

    # One delta for every listed body, so stacked objects stay together. Retry offsets that
    # would wedge an object into the counter/microwave -- MuJoCo ejects those violently.
    for attempt in range(max_tries):
        for b, (a, q) in saved.items():
            sim.data.qpos[a:a + 7] = q
        sim.forward()
        for g, rng_xy in groups:                     # one delta per group
            # a scalar is symmetric in both axes; a pair is (x_range, y_range).
            rx, ry = rng_xy if isinstance(rng_xy, (tuple, list)) else (rng_xy, rng_xy)
            dx = rng.uniform(-rx, rx)
            dy = rng.uniform(-ry, ry)
            d_theta = rng.uniform(-cfg["obj_yaw"], cfg["obj_yaw"]) if cfg["obj_yaw"] else 0.0
            for body in g:
                change_object_position(sim, body, dx=dx, dy=dy, dz=0.0, d_theta=d_theta)
        if not _intersecting(sim, geoms):
            break
    else:
        for b, (a, q) in saved.items():
            sim.data.qpos[a:a + 7] = q
        sim.forward()
        print(f"[randomize] {task}: no clear object placement in {max_tries} tries, left unjittered")

    back = rng.uniform(*cfg["base_back"])
    change_robot_position(
        sim, "mobilebase0_base",
        dx=_BASE_BACK_SIGN * back,
        dy=rng.uniform(-cfg["base_side"], cfg["base_side"]),
        dz=0.0,
        d_theta=rng.uniform(-cfg["base_yaw"], cfg["base_yaw"]),
    )
