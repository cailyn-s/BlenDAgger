"""Report the components of a composite task's success check, not just their AND.

robocasa computes each sub-condition inside `_check_success` and then collapses them into a
single bool, so a rollout that got the food into the microwave but never pressed the button
is indistinguishable from one that did nothing. robomimic's `is_success()` already supports
returning `{"task": bool, ...}` with extra keys (env_robosuite.py), and run_rollout already
forwards those keys into its results -- the only missing piece is the tasks returning a dict.

Safe to apply globally: everything in this pipeline reads `is_success()["task"]`
(collector.py, video_rollout.py), robosuite's done/reward path never calls
`_check_success`, and Kitchen.reward() returns 0 unconditionally. robocasa's own standalone
scripts (collect_demos.py, eval_utils.py) do `if env._check_success():` and would see a
truthy dict, but this pipeline does not use them.
"""

SUBTASKS = {
    # Cumulative milestones, not independent conditions: each stage includes the one before it,
    # so the three numbers read as a funnel (how far down the task the policy gets).
    # *_at_end is NOT latched: run_rollout records the last timestep's value for these, so
    # "mug ever placed" and "mug still placed when the episode ended" are both reported.
    "PrepareCoffee": ["s1_mug_under_dispenser", "s2_mug_and_button", "s3_full_success",
                      "s1_mug_under_dispenser_at_end"],
    # Cumulative funnel s1->s4 plus the raw conditions. Note microwave.update_state() forces
    # _turned_on=False whenever the door is open, so microwave_on already implies the door is
    # shut -- s4's door term is redundant but kept so the funnel reads explicitly.
    "MicrowaveThawing": ["microwave_door_open", "food_in_microwave", "released_food",
                         "microwave_on", "s3_food_in_and_closed", "s4_food_in_closed_and_on"],
    "ArrangeVegetables": ["s1_vegetable1_on_board", "s2_both_on_board", "s3_full_success"],
}


def patch_subtask_success():
    import robocasa.utils.object_utils as OU
    from robocasa.environments.kitchen.multi_stage.brewing.prepare_coffee import PrepareCoffee
    from robocasa.environments.kitchen.multi_stage.defrosting_food.microwave_thawing import (
        MicrowaveThawing,
    )
    from robocasa.environments.kitchen.multi_stage.chopping_food.arrange_vegetables import (
        ArrangeVegetables,
    )

    def prepare_coffee_check(self):
        # mirrors PrepareCoffee._check_success exactly; only the return shape differs
        mug_under_dispenser = self.coffee_machine.check_receptacle_placement_for_pouring(self, "obj")
        released_mug = OU.gripper_obj_far(self)
        machine_on = bool(self.coffee_machine._turned_on)
        gripper_clear_of_button = self.coffee_machine.gripper_button_far(self)
        full = bool(mug_under_dispenser and released_mug and machine_on and gripper_clear_of_button)
        # Each stage is evaluated per timestep and latched by run_rollout, so s2 requires the mug
        # to be placed AND the machine on at the same moment -- not each at some point.
        return {
            "task": full,
            "s1_mug_under_dispenser": bool(mug_under_dispenser),
            "s2_mug_and_button": bool(mug_under_dispenser and machine_on),
            "s3_full_success": full,
            # last-timestep value, not latched -- distinguishes a policy that places the mug
            # and leaves it there from one that places it and then knocks it out
            "s1_mug_under_dispenser_at_end": bool(mug_under_dispenser),
        }

    def microwave_thawing_check(self):
        # mirrors MicrowaveThawing._check_success exactly for "task"; the rest are breakdowns
        food_in_microwave = OU.obj_inside_of(self, "obj", self.microwave)
        microwave_on = bool(self.microwave.get_state()["turned_on"])
        released_food = OU.gripper_obj_far(self)
        # 0 = shut, 1 = fully open. 0.005 is robocasa's own threshold for this fixture
        # (see heat_mug.py _check_success and microwave.update_state). Not reported on its
        # own: the door starts closed, so a latched "ever closed" is 1.0 for every rollout.
        door_state = self.microwave.get_door_state(self)["door"]
        door_closed = bool(door_state <= 0.005)
        # Latched, so this reads "did the policy ever get the door open" -- the first real
        # milestone. The inverse is useless here: the door starts closed, so "ever closed"
        # is 1.0 for every rollout.
        # 0.3 of the [0, pi/2] hinge range = ~27 degrees: open enough to be deliberate and to
        # admit the food, without demanding robocasa's 0.90 "fully opened" bar. Deliberately
        # stricter than the 0.005 interlock used for door_closed above, so the two are not
        # complements: between 0.005 and 0.3 the door counts as neither.
        door_open = bool(door_state > 0.3)
        return {
            "task": bool(food_in_microwave and released_food and microwave_on),
            "microwave_door_open": door_open,
            "food_in_microwave": bool(food_in_microwave),
            "released_food": bool(released_food),
            "microwave_on": microwave_on,
            "s3_food_in_and_closed": bool(food_in_microwave and door_closed),
            "s4_food_in_closed_and_on": bool(food_in_microwave and door_closed and microwave_on),
        }

    def arrange_vegetables_check(self):
        # mirrors ArrangeVegetables._check_success exactly
        veg1 = OU.check_obj_in_receptacle(self, "vegetable1", "cutting_board")
        veg2 = OU.check_obj_in_receptacle(self, "vegetable2", "cutting_board")
        gripper_clear = OU.gripper_obj_far(self, obj_name="cutting_board")
        full = bool(veg1 and veg2 and gripper_clear)
        # Cumulative milestones, same shape as PrepareCoffee: each stage contains the previous,
        # evaluated per timestep and latched by run_rollout (so s2 needs both on the board at once).
        return {
            "task": full,
            "s1_vegetable1_on_board": bool(veg1),
            "s2_both_on_board": bool(veg1 and veg2),
            "s3_full_success": full,
        }

    ArrangeVegetables._check_success = arrange_vegetables_check
    PrepareCoffee._check_success = prepare_coffee_check
    MicrowaveThawing._check_success = microwave_thawing_check
    return sorted(SUBTASKS)
