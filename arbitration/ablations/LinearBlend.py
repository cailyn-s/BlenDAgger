from arbitration.ActionArbitrator import ActionArbitrator

# Fixed-alpha ablation: blends pose actions with a constant alpha; the gripper always follows the human.

class LinearBlend(ActionArbitrator):
    def __init__(self, alpha=0.5, train_all_steps=True):
        super().__init__()
        self.alpha = alpha
        self.train_all_steps = train_all_steps  # False trains on human interventions only

    def reset(self):
        pass

    def arbitrate(self, *, robot_action, human_action, human_active, **_):
        if not human_active:
            return {
                "action": robot_action,
                "control_mode": "robot",
                "intervening": False,
                "training" : self.train_all_steps
            }

        blended = robot_action.copy()

        # Orientation is an axis-angle delta, so linear blending is a reasonable approximation.
        blended[0:8] = (
            (1.0 - self.alpha) * human_action[0:8]
            + self.alpha * robot_action[0:8]
        )
        # always use human gripper action
        blended[6] = human_action[6]

        return {
            "action": blended,
            "control_mode": "blend",
            "intervening": True,
            "training": True,
            "alpha": self.alpha,
            "metadata": {"algo": "LinearBlend", "alpha": self.alpha}
        }
