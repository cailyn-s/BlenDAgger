import os

import cv2
import numpy as np

WINDOW = "Policy Multi-View"

# MuJoCo fixes vertical FOV and derives horizontal FOV from width/height, so aspect sets how much scene is visible.
# The FOV must match the collected data; override with VIEWER_PANE=WxH (e.g. 960x720).
PANE_W, PANE_H = 940, 529
if os.environ.get("VIEWER_PANE"):
    PANE_W, PANE_H = (int(v) for v in os.environ["VIEWER_PANE"].lower().split("x"))


class MultiViewRenderer:
    def __init__(self, robomimic_env, height=PANE_H, width=PANE_W):
        # robomimic EnvRobosuite
        self.robomimic_env = robomimic_env

        # raw robosuite env
        self.rs_env = robomimic_env.env

        self.height = height
        self.width = width

        self.cameras = [
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]

        self.key_pressed = False
        self.key_timer = 0
        self._window_ready = False

    def _ensure_window(self):
        # Lazy: browse_fixed_envs.py constructs this just to read .cameras and must not
        # get a window. WINDOW_NORMAL so an oversized combined image is scaled, not clipped.
        if self._window_ready:
            return
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.width * len(self.cameras), self.height)
        self._window_ready = True

    def render(self):
        self._ensure_window()
        images = []

        for cam in self.cameras:
            # MuJoCo offscreen render
            img = self.rs_env.sim.render(
                camera_name=cam,
                height=self.height,
                width=self.width,
            )

            img = cv2.flip(img, 0) # Flip vertically (MuJoCo renders upside down)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Converts RGB -> BGR safely

            if self.key_pressed:
                h, w = img.shape[:2]
                cv2.rectangle(
                    img,
                    (0, 0),
                    (w - 1, h - 1),
                    (0, 165, 255), # orange
                    6
                )

            images.append(img)

        # Concatenate horizontally
        combined = np.concatenate(images, axis=1)
        cv2.imshow(WINDOW, combined)

        key = cv2.waitKey(1)
        if key != -1:
            self.key_timer = 5

        if self.key_timer > 0:
            self.key_timer -= 1
            self.key_pressed = True
        else:
            self.key_pressed = False

    def close(self):
        cv2.destroyAllWindows()
        # Collector calls this after every episode. Without clearing the latch the next
        # episode's imshow re-creates the window itself, with the WINDOW_AUTOSIZE default.
        self._window_ready = False