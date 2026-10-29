import numpy as np
from arbitration.ActionArbitrator import ActionArbitrator

class HGDagger(ActionArbitrator):
    def __init__(
        self,
        train_rollouts=True,           # train on autonomous rollout steps (default True)
        button_for_intervention=False, # require spacebar press to end intervention
        train_all_steps=False,         # alias for train_rollouts
        ema_alpha=0.5,                 # onset EMA decay (0.0=off; default 0.5 = ~7 steps to <1%)
        idle_release_s=0.0,            # hand back after this long with no device input (0.0=off)
        control_hz=30.0,               # loop rate used to convert idle_release_s into steps
        human_active_threshold=0.0,    # device deadzone for engagement (0.0=off: trust caller's human_active)
    ):
        super().__init__()
        # either flag enables training on rollout steps
        self.train_rollouts = train_rollouts or train_all_steps
        self.button_for_intervention = button_for_intervention
        self.human_active_threshold = float(human_active_threshold)
        self.ema_alpha = ema_alpha
        self.idle_release_s = float(idle_release_s)
        self.control_hz = float(control_hz) or 30.0
        # 0 disables the dwell entirely, restoring the button-only handback.
        self._idle_release_steps = (max(1, round(self.idle_release_s * self.control_hz))
                                    if self.idle_release_s > 0.0 else 0)
        self._invalidates_robot_plan = True   # full takeover invalidates the robot's queued chunk (collector clears it; blends leave it False)
        self._prev_action = None        # last executed action; anchor for onset blend
        self._onset_step = None         # steps into onset window; None = not blending
        self._prev_human_active = False
        self._idle_steps = 0            # consecutive steps with no device input
        self._prev_use_human = False    # whether the human held control last step

    def reset(self):
        self._prev_action = None
        self._onset_step = None
        self._prev_human_active = False
        self._idle_steps = 0
        self._prev_use_human = False

    def arbitrate(self, *, robot_action, human_action, human_active, intervening,
                  gripper_active=False, **_):
        if self.human_active_threshold > 0.0:
            # Engagement off the device deadzone instead of the caller's 1e-6 noise floor.
            # Whole-action test, no DoF grouping: any axis past the deadzone hands every DoF
            # to the human, because HG-DAgger is a hard switch, not a per-DoF blend.
            ha = np.asarray(human_action, dtype=np.float64)
            human_active = bool(np.any(np.abs(ha[:6]) > self.human_active_threshold)
                                or gripper_active)

        self._idle_steps = 0 if human_active else self._idle_steps + 1

        if self._idle_release_steps > 0:
            # Idle dwell owns the handback: the human keeps control until the device
            # has been quiet for the dwell. Clearing the device latch (the release
            # button) still hands back immediately, it is just no longer required.
            use_human_action = human_active or (
                intervening and self._prev_use_human
                and self._idle_steps < self._idle_release_steps
            )
        else:
            use_human_action = human_active or (intervening and self.button_for_intervention)

        if use_human_action:
            action = human_action
            training = True
        else:
            action = robot_action
            training = self.train_rollouts

        # Robot authority alpha for logging, same convention/format as the blend
        # arbitrators: a 6D per-DoF vector (pose only, gripper excluded) where
        # 1 = full robot, 0 = full human. HG-DAgger is a hard switch, so alpha is
        # 0 or 1, except during the onset EMA window where the robot's decaying
        # residual weight (ema_alpha ** (onset_step + 1)) lifts it slightly off 0.
        alpha = 1.0 if not use_human_action else 0.0

        if self.ema_alpha > 0.0:
            if not use_human_action:
                # Robot step: keep anchor fresh; no onset window active.
                self._prev_action = action.copy()
                self._onset_step = None
            elif intervening and not human_active:
                # Idle intervention: clear onset window.
                self._onset_step = None
            else:
                # Active human step.
                if not self._prev_human_active:
                    # First step of takeover: open the onset window.
                    self._onset_step = 0

                if self._onset_step is not None and self._prev_action is not None:
                    # Rolling EMA: robot contribution decays as alpha^(step+1).
                    # Stop once it drops below 1% — human has full control after that.
                    robot_weight = self.ema_alpha ** (self._onset_step + 1)
                    if robot_weight >= 0.01:
                        smoothed = action.copy()
                        smoothed[:6] = (self.ema_alpha * self._prev_action[:6]
                                        + (1.0 - self.ema_alpha) * action[:6])
                        action = smoothed
                        self._prev_action = action.copy()
                        self._onset_step += 1
                        alpha = robot_weight    # this step's decaying robot authority
                    else:
                        self._onset_step = None  # decay complete

        self._prev_human_active = human_active
        self._prev_use_human = use_human_action

        return {
            "action": action,
            "control_mode": "human" if use_human_action else "robot",
            "intervening": bool(use_human_action),
            "training": training,
            "alphas": np.full(6, alpha),
            "metadata": {"algo": "HGDagger", "train_rollouts": self.train_rollouts,
                         "button_for_intervention": self.button_for_intervention,
                         "ema_alpha": self.ema_alpha,
                         "idle_release_s": self.idle_release_s,
                         "human_active_threshold": self.human_active_threshold},
        }
