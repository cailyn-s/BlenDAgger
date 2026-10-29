# Based off of Robocasa collect_demos
from copy import deepcopy
import time

import numpy as np
import robosuite

from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from collection.teleop.KeyboardHoldGripper import KeyboardHoldGripper

class Teleop:
    def __init__(
        self,
        env,
        device,
        mirror_actions=True,
        max_fr=30.0,
    ):
        self.env = env
        self.device = device
        self.mirror_actions = mirror_actions
        self.max_fr = max_fr # loop rate is controlled in the collector
        self.intervening = False
    
    def reset(self):
        self.intervening = False
        self.device.reset()

    def get_device_teleop_action(self, prev_action):        
        # Active robot
        active_robot = self.env.robots[self.device.active_robot]

        # Read input
        input_ac_dict = self.device.input2action(mirror_actions=self.mirror_actions)

        # If reset key pressed, return None
        if input_ac_dict is None:
            self.intervening = False
            return None, None, False

        action_dict = deepcopy(input_ac_dict)

        # Convert delta / absolute inputs to controller-compatible format
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError(f"Unknown controller input type: {controller_input_type}")
        
        # Check whether gripper is being controlled (otherwise set to previous gripper value)
        gripper_key = f"{arm}_gripper"
        if hasattr(self.device, "gripper_command"):
            if self.device.gripper_command is None:
                # Gripper key isn't held, so fallback to the last known state (or open if no previous state)
                if prev_action is not None:
                    action_dict[gripper_key] = np.array([prev_action[6]]) # gripper is index 6
                else:
                    action_dict[gripper_key] = np.array([1.0]) # default to open
            else:
                action_dict[gripper_key] = np.array([self.device.gripper_command])

        env_action = active_robot.create_action_vector(action_dict)

        # Check if they are done intervening
        if self.device.done_intervening:
            self.intervening = False
            self.device.done_intervening = False
        if self.has_nonzero_action(action_dict):
            self.intervening = True

        return env_action, action_dict, self.intervening

    def has_gripper_action(self, action_dict):
        # Gripper action (hold-based)
        if hasattr(self.device, "gripper_command"):
            if self.device.gripper_command is not None:
                return True

    def has_nonzero_action(
        self,
        curr_action_dict,
        tol=1e-6,
    ):
        """
        Returns True if any arm action is non-zero or a gripper key is held (gripper_command is not None)
        """
        for arm in self.env.robots[0].arms:
            # Arm motion
            delta_key = f"{arm}_delta"
            if delta_key in curr_action_dict:
                delta = curr_action_dict[delta_key]
                if np.any(np.abs(delta) > tol):
                    return True
        if self.has_gripper_action(curr_action_dict):
            return True

        return False

    def get_action(self, obs, robot_action, prev_action):
        start = time.time()

        env_action, action_dict, intervening = self.get_device_teleop_action(prev_action)

        if env_action is None:
            return None, None, False

        return env_action, action_dict, intervening
