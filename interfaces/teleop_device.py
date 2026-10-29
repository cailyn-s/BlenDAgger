from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class AbstractTeleopDevice(ABC):
    """
    Uniform interface for all teleoperation input devices (SpaceMouse, keyboard, etc.).

    Output convention: 7D EEF-delta vector
        [dx, dy, dz, droll, dpitch, dyaw, gripper]

    get_action returns a 3-tuple:
      action_7d    — the commanded action (None signals episode abort)
      human_active — True when the action has significant nonzero magnitude
                     this step (used by arbitrators to weight blending)
      is_intervening — True when the human has declared themselves in control
                       (may remain True even when not actively moving, e.g.
                       after first touch until spacebar release)
    """

    @abstractmethod
    def get_action(
        self, obs: dict, prev_action: np.ndarray
    ) -> tuple[np.ndarray | None, bool, bool]:
        """
        Read device state and return (action_7d, human_active, is_intervening).
        Return action_7d=None to request episode abort (e.g. hold both buttons).
        """
        raise NotImplementedError

    @property
    def grip_held(self) -> bool:
        """True when the device's motion gate (e.g. grip trigger) is currently held.
        Devices without such a gate always return True."""
        return True

    def reset(self) -> None:
        """Optional: reset device state at episode start."""
        pass


class NullTeleopDevice(AbstractTeleopDevice):
    """
    Returns all-zero actions with human_active=False, is_intervening=False.
    Used in evaluation mode so the same collection loop runs with no human channel.
    """

    def get_action(
        self, obs: dict, prev_action: np.ndarray
    ) -> tuple[np.ndarray, bool, bool]:
        return np.zeros(7, dtype=np.float32), False, False
