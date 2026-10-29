from arbitration.CombinedBlend import CombinedBlend

class CombinedBlend_NoSimilarity(CombinedBlend):
    """
    Ablation of CombinedBlend with no use of similarity.
    Blending is driven by confidence (uncertainty) and magnitude alone; the
    magnitude gate and smoothing are inherited from CombinedBlend.arbitrate().
    alpha_sim is reported as 1.0 in the diagnostics (its neutral value).
    """

    def _compute_alpha_target(self, similarity, confidence, magnitude):
        if confidence is not None:
            alpha_conf = self.compute_sigmoid_alpha(confidence, magnitude, self.slope_conf, self.offset_conf)
        else:
            alpha_conf = 1.0
        return alpha_conf, 1.0, alpha_conf
