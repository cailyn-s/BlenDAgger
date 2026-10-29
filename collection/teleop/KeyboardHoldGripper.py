from robosuite.devices import Keyboard
from pynput.keyboard import Key

class KeyboardHoldGripper(Keyboard):
    """
    Keyboard device with hold-based gripper control.

    - Key '4' : open gripper (hold; command -1.0)
    - Key '3' : close gripper (hold; command +1.0)

    All other behavior is identical to robosuite Keyboard.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gripper_command = None  # -1.0 (open) or 1.0 (close), robosuite convention; None when no key held
        self.done_intervening = False
    
    def reset(self):
        """Reset custom states on environment reset."""
        self.gripper_command = None
        self.done_intervening = False
        self._reset_state = 0
        self._enabled = True
    
    def close(self):
        pass

    def on_press(self, key):
        """
        Key handler for key presses.
        Extends base behavior with hold-based gripper control.
        """
        # Run the original on_press logic (motion, rotation, etc.)
        super().on_press(key)

        try:
            # Hold-based gripper OPEN
            if key.char == "4":
                self.gripper_command = -1.0
                self.grasp_states[self.active_robot][self.active_arm_index] = True

            # Hold-based gripper CLOSE
            elif key.char == "3":
                self.gripper_command = 1.0
                self.grasp_states[self.active_robot][self.active_arm_index] = False

        except AttributeError:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Releasing '3'/'4' clears the gripper command; shift release ends the intervention.
        """
        try:
            # Releasing shift ends the intervention until another key is pressed
            if key == Key.shift or key == Key.shift_r:
                self.done_intervening = True
                return
            else:
                self.done_intervening = False
            
            # Releasing a gripper key stops commanding the gripper
            if key.char in ("3", "4"):
                self.gripper_command = None

            # controls for mobile base (only applicable if mobile base present)
            elif key.char == "b":
                self.base_modes[self.active_robot] = not self.base_modes[self.active_robot]

            # user-commanded reset
            elif key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()

            elif key.char == "s":
                self.active_arm_index = (
                    self.active_arm_index + 1
                ) % len(self.all_robot_arms[self.active_robot])

            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots

        except AttributeError:
            pass
