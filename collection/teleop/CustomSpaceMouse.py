import os
import time
import numpy as np
from pynput import keyboard
from robosuite.devices import SpaceMouse
from robosuite.devices.spacemouse import convert

class CustomSpaceMouse(SpaceMouse):
    """
    Custom SpaceMouse Mapping:
    - Left Button (press): Latch Gripper Close (1.0)
    - Right Button (press): Latch Gripper Open (-1.0)
    - Spacebar (keyboard): Release Control to Robot Policy
    - Both Buttons (Hold > 1s): Reset Environment

    stale_timeout (or BLENDAGGER_SPACEMOUSE_STALE_S): if no HID packet arrives within this
    time, motion is zeroed instead of replaying the last packet (e.g. for a puck
    forwarded over the network). Disabled by default.
    """
    def __init__(self, *args, stale_timeout=None, **kwargs):
        self.active_robot = 0
        self.active_arm_indices = [0]
        self.base_modes = [False]
        
        # Custom states
        self.gripper_command = None
        self.done_intervening = False 
        self._ignore_until_clear = False
        
        # Tap/Hold timer states
        self._both_pressed_time = None
        self._hold_threshold = 1.0  
        
        # Debounce timer states
        self._single_press_time = None
        self._debounce_window = 0.05  # 50ms grace period to detect the second button
        self._pending_gripper = None

        # HID freshness — set before super() since it starts the reader thread.
        self._stale_timeout = float(os.environ.get("BLENDAGGER_SPACEMOUSE_STALE_S", 0.0)) if stale_timeout is None else float(stale_timeout)
        self._last_packet_t = None
        self._packet_count = 0
        self._stale_trips = 0

        # Spacebar listener
        self._space_listener = keyboard.Listener(on_press=self._on_key_press)
        self._space_listener.start()
        
        super().__init__(*args, **kwargs)
        
        # Initialize custom space mouse control
        self.start_control()
        print("[CustomSpaceMouse] Initialized and forced start_control(). Ready for input!")
    
    def _on_key_press(self, key):
        if key == keyboard.Key.space:
            if not getattr(self, 'done_intervening', False):
                self.done_intervening = True
                print("[SpaceMouse] Control released to policy.")

    def reset(self):
        """Reset custom states on environment reset."""
        self.gripper_command = None
        self.done_intervening = False
        self._ignore_until_clear = False
        self._both_pressed_time = None
        self._single_press_time = None
        self._pending_gripper = None
        self._reset_state = 0
        self._enabled = True
        self._stale_trips = 0  # per-episode counter

    def close(self):
        """Cleanly shuts down the background thread and releases the hardware lock."""
        self._enabled = False  # Signal the run() loop to stop processing
        
        if hasattr(self, '_space_listener'):
            self._space_listener.stop()
            
        if hasattr(self, 'device') and self.device is not None:
            try:
                self.device.close()
                print("[SpaceMouse] Hardware lock released successfully.")
            except Exception as e:
                print(f"[SpaceMouse] Warning - Could not close device: {e}")

    @property
    def control(self):
        """6-DoF pose, zeroed once the HID stream goes silent (no-op unless stale_timeout > 0)."""
        if self._stale_timeout > 0.0 and self._last_packet_t is not None:
            if time.time() - self._last_packet_t > self._stale_timeout and any(self._control):
                self._stale_trips += 1
                self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.x = self.y = self.z = 0
                self.roll = self.pitch = self.yaw = 0
        return np.array(self._control)

    def input_age(self):
        """Seconds since the last HID packet, or None before the first one arrives."""
        return None if self._last_packet_t is None else time.time() - self._last_packet_t

    def input_stats(self):
        """HID transport health snapshot for the per-episode timing summary."""
        return {
            "packets": self._packet_count,
            "stale_trips": self._stale_trips,
            "stale_timeout_s": self._stale_timeout,
        }

    @property
    def control_gripper(self):
        """Returns 1.0 for close, -1.0 for open, or None for neutral."""
        # Process the debounce check here since teleop.py polls this continuously
        if self._single_press_time is not None and self._pending_gripper is not None:
            if time.time() - self._single_press_time >= self._debounce_window:
                self.gripper_command = self._pending_gripper
                try:
                    self.grasp_states[self.active_robot][self.active_arm_indices[0]] = (self.gripper_command == -1.0)
                except AttributeError:
                    pass
        return self.gripper_command

    def run(self):
        """Overrides the parent's background listener to implement custom key press handling."""
        while True:
            try:
                d = self.device.read(13)
            except Exception as e:
                continue
                
            if d is not None and len(d) > 0 and self._enabled:
                current_time = time.time()
                self._last_packet_t = current_time
                self._packet_count += 1

                # Check for puck movement or button presses
                is_pressing_buttons = (d[0] == 3 and len(d) >= 2 and (d[1] & 3) > 0)
                is_moving = any(abs(v) > 0.01 for v in getattr(self, '_control', [0]*6))
                
                if is_moving or is_pressing_buttons:
                    self.done_intervening = False
                
                # Evaluate hold threshold for resetting
                if self._both_pressed_time is not None:
                    if current_time - self._both_pressed_time > self._hold_threshold:
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()
                        self.gripper_command = None
                        self._both_pressed_time = None
                        print("[SpaceMouse] Reset triggered via Hold.")

                # 6-DoF Processing
                if self.product_id == 50741:
                    if d[0] == 1 and len(d) >= 7:
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0
                    elif d[0] == 2 and len(d) >= 7:
                        self.roll = convert(d[1], d[2])
                        self.pitch = convert(d[3], d[4])
                        self.yaw = convert(d[5], d[6])
                        self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
                else:
                    if d[0] == 1 and len(d) >= 13:
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0
                        self.roll = convert(d[7], d[8])
                        self.pitch = convert(d[9], d[10])
                        self.yaw = convert(d[11], d[12])
                        self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

                # Button and intervention processing
                if d[0] == 3 and len(d) >= 2:
                    left_pressed = (d[1] & 1) > 0
                    right_pressed = (d[1] & 2) > 0
                    both_pressed = left_pressed and right_pressed

                    # Clear the ignore flag once all buttons are fully released
                    if not left_pressed and not right_pressed:
                        self._ignore_until_clear = False

                    # Both buttons logic (Hold to reset)
                    if both_pressed:
                        self._ignore_until_clear = True # Lock out single-button gripper logic
                        self._single_press_time = None  # Cancel any pending single-button actions
                        
                        if self._both_pressed_time is None:
                            self._both_pressed_time = current_time 
                    else:
                        if self._both_pressed_time is not None:
                            self._both_pressed_time = None
                            
                        # Hold-based gripper mapping
                        # Skip if we are trailing off a "both buttons" press
                        if self._ignore_until_clear:
                            self.gripper_command = None
                            continue
                            
                        # Wrap in try/except to prevent silent thread crashes on initialization
                        try:
                            if left_pressed or right_pressed:
                                # Single press
                                if self._single_press_time is None:
                                    self._single_press_time = current_time
                                    self._pending_gripper = 1.0 if left_pressed else -1.0
                            else:
                                # Neither pressed
                                self._single_press_time = None
                                self._pending_gripper = None
                                self.gripper_command = None  
                        except AttributeError:
                            pass