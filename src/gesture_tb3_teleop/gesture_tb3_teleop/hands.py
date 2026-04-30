#hands.py
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self,
                 max_num_hands=1,
                 det_conf=0.6,
                 track_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        self.drawer = mp.solutions.drawing_utils
        self.drawer_style = mp.solutions.drawing_styles

    def process(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw(self, bgr_frame, hand_landmarks):
        self.drawer.draw_landmarks(
            bgr_frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.drawer_style.get_default_hand_landmarks_style(),
            self.drawer_style.get_default_hand_connections_style()
        )

    @staticmethod
    def to_xy_array(hand_landmarks, img_shape):
        """Return a 21x2 pixel-coordinate array (x, y) and a 21x2 normalized-coordinate array (0–1)."""
        import numpy as np
        h, w = img_shape[:2]
        xs, ys = [], []
        for lm in hand_landmarks.landmark:
            xs.append(lm.x * w)
            ys.append(lm.y * h)
        xs = np.array(xs)
        ys = np.array(ys)
        pix = np.stack([xs, ys], axis=1)  # 21x2 pixel coordinates
        norm = np.stack([xs / w, ys / h], axis=1)  # 21x2 normalized coordinates
        return pix, norm
