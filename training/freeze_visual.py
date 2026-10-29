import robomimic.algo.bc as bc

def patch_freeze_visual_encoder():

    original_init = bc.BC.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        policy = self.nets["policy"]
        encoder = policy.nets["encoder"]

        for p in encoder.parameters():
            p.requires_grad = False
        print("Frozen policy encoder")

    bc.BC.__init__ = patched_init