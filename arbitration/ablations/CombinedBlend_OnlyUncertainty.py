from arbitration.ablations.CombinedBlend_SingleGroup import CombinedBlend_SingleGroup


class CombinedBlend_OnlyUncertainty(CombinedBlend_SingleGroup):
    """
    Single-component variant of CombinedBlend: confidence is the ONLY blending signal.

    alpha = sigmoid(slope_conf * (confidence - offset_conf))

    Everything else the base blend uses to set authority is removed:
      * no similarity term — alpha_sim is reported as 1.0;
      * no magnitude term in the sigmoid offset (m = 0), no magnitude gate and no
        exact-zero guard, so how hard the human pushes never affects authority;
      * one merged blend group over all 6 EEF dims (inherited from CombinedBlend_SingleGroup), so the
        mean normalised uncertainty over every EEF dim produces one alpha.

    Temporal smoothing, the grace period and the blend modes are inherited from
    CombinedBlend.arbitrate() unchanged. An uncertainty estimator and calibration
    file are required; with no confidence signal alpha falls back to 1.0 (full robot).

    Note: with the gate removed this is the one variant whose authority is
    independent of the human's input. While the policy is uncertain the human is
    given authority even on the frames they are not driving, so the blended command
    shrinks toward the (near-zero) human action and the arm slows.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.magnitude_gate = False
        self.zero_input_guard = False

    def _compute_alpha_target(self, similarity, confidence, magnitude):
        if confidence is not None:
            alpha_conf = self.compute_sigmoid_alpha(confidence, 0.0, self.slope_conf, self.offset_conf)
        else:
            alpha_conf = 1.0
        return alpha_conf, 1.0, alpha_conf
