from arbitration.CombinedBlend import CombinedBlend

class CombinedBlend_NoMagnitude(CombinedBlend):
    """
    Ablation of CombinedBlend with no adjustment based on magnitude.

    The magnitude term in the sigmoid offsets is set to 0.0, so how hard the
    human pushes has no effect, and the idle-group guard is disabled.
    Everything else (normalisation, similarity, confidence, smoothing, grace
    period) is inherited from CombinedBlend.arbitrate().
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.magnitude_gate = False
        self.zero_input_guard = False

    def _compute_alpha_target(self, similarity, confidence, magnitude):
        alpha_sim = self.compute_sigmoid_alpha(similarity, 0.0, self.slope_sim, self.offset_sim)
        if confidence is not None:
            alpha_conf = self.compute_sigmoid_alpha(confidence, 0.0, self.slope_conf, self.offset_conf)
        else:
            alpha_conf = 1.0
        return alpha_sim * alpha_conf, alpha_sim, alpha_conf
