import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool

# --- Reuse your EC720 modules (copied into this ROS package) ---
from .hands import HandDetector
from .features import pack_features
from .rules import classify_gesture


# ====== EC720-aligned feature order (same as infer_use_svm.py) ======
FEATURE_ORDER = [
    "palm_size",
    "curl_index", "curl_middle", "curl_ring", "curl_pinky",
    "thumb_straight", "thumb_lateral",
    "yaw_norm",
]

def feats_to_vector(feats: dict) -> np.ndarray:
    """Convert pack_features dict into a vector according to FEATURE_ORDER (missing keys filled with 0)."""
    return np.array([feats.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float32).reshape(1, -1)

def load_svm(model_path: Path):
    """Load SVM model; return (clf, has_proba)."""
    try:
        import joblib
        if model_path.exists():
            clf = joblib.load(model_path)
            has_proba = hasattr(clf, "predict_proba")
            return clf, has_proba
    except Exception as e:
        print(f"[WARN] Load SVM failed: {e}")
    return None, False

def fuse_labels(label_rule: str, label_model: str, model_prob: float, prob_th: float = 0.6) -> str:
    """
    EC720 fusion strategy:
    - If model exists and its predicted probability >= prob_th, use model label;
    - Otherwise, fall back to rule label.
    """
    if label_model is not None and model_prob is not None and model_prob >= prob_th:
        return label_model
    return label_rule


class GestureCmdVel(Node):
    def __init__(self):
        super().__init__('gesture_cmd_vel')

        # ====== ROS parameters (names aligned to EC720 where applicable) ======
        # Camera & display
        self.declare_parameter('cam', 0)                 # == EC720 --cam
        self.declare_parameter('mirror', True)           # EC720 uses cv2.flip(frame, 1)
        self.declare_parameter('show_camera', True)      # show OpenCV window

        # MediaPipe Hands detector params (hands.py defaults)
        self.declare_parameter('max_num_hands', 1)
        self.declare_parameter('det_conf', 0.6)
        self.declare_parameter('track_conf', 0.6)

        # SVM
        self.declare_parameter('model', 'models/svm.pkl')  # == EC720 --model (relative)
        self.declare_parameter('svm_th', 0.60)             # == EC720 --svm_th
        self.declare_parameter('use_svm', True)

        # EC720 temporal logic
        self.declare_parameter('smooth_k', 5)                # == SMOOTH_K
        self.declare_parameter('lock_frames', 3)             # == LOCK_FRAMES
        self.declare_parameter('switch_confirm_frames', 2)   # == SWITCH_CONFIRM_FRAMES

        # CmdVel scaling (safe defaults for TB3 burger in VM)
        self.declare_parameter('lin_speed', 0.20)   # m/s
        self.declare_parameter('ang_speed', 1.00)   # rad/s

        # ====== Read params ======
        cam_index = int(self.get_parameter('cam').value)
        self.mirror = bool(self.get_parameter('mirror').value)
        self.show_camera = bool(self.get_parameter('show_camera').value)

        max_num_hands = int(self.get_parameter('max_num_hands').value)
        det_conf = float(self.get_parameter('det_conf').value)
        track_conf = float(self.get_parameter('track_conf').value)

        self.use_svm = bool(self.get_parameter('use_svm').value)
        self.svm_th = float(self.get_parameter('svm_th').value)

        self.smooth_k = int(self.get_parameter('smooth_k').value)
        self.lock_frames = int(self.get_parameter('lock_frames').value)
        self.switch_confirm_frames = int(self.get_parameter('switch_confirm_frames').value)

        self.lin_speed = float(self.get_parameter('lin_speed').value)
        self.ang_speed = float(self.get_parameter('ang_speed').value)

        # ====== ROS publisher ======
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_gesture = self.create_publisher(String, '/gesture/stable', 10)
        self.pub_mode    = self.create_publisher(String, '/gesture/mode', 10)
        self.pub_estop   = self.create_publisher(Bool,   '/gesture/estop', 10)
	
        # ====== Camera ======
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera open failed (index={cam_index}). "
                f"If in VM: VirtualBox window -> Devices -> Webcams -> enable your camera."
            )

        # ====== Hand detector (EC720 hands.py defaults) ======
        self.det = HandDetector(
            max_num_hands=max_num_hands,
            det_conf=det_conf,
            track_conf=track_conf
        )

        # ====== SVM model ======
        self.clf, self.has_proba = (None, False)
        if self.use_svm:
            # model path relative to this module directory
            pkg_dir = Path(__file__).resolve().parent
            model_rel = str(self.get_parameter('model').value)
            model_path = (pkg_dir / model_rel).resolve()
            self.clf, self.has_proba = load_svm(model_path)
            if self.clf is None:
                self.get_logger().warn("SVM model not found/failed. Running RULE-only mode.")
            else:
                self.get_logger().info(f"SVM loaded: {model_path} | has_proba={self.has_proba}")
        else:
            self.get_logger().info("use_svm:=false -> RULE-only mode.")

        # ====== EC720 voting + hysteresis state ======
        self.history = deque(maxlen=self.smooth_k)
        self.candidate = None
        self.cand_count = 0

        # ====== EC720 mode state machine ======
        self.mode = "IDLE"         # IDLE / NAV
        self.last_locked = "None"  # previous locked stable gesture
        self.label_stable = "None" # current locked stable gesture

        # FPS monitor (optional)
        self.fps_t0, self.fps_cnt = time.time(), 0
        self.fps_disp = 0.0

        self.get_logger().info("GestureCmdVel started. ESC in camera window to exit. Ctrl+C in terminal to stop.")

    def map_to_twist(self, gesture: str) -> Twist:
        """
        Gesture -> cmd_vel mapping (keep consistent with your EC720 semantics):
          - OpenPalm  : forward
          - ThumbUp   : backward
          - PointLeft : turn left  (+angular.z)
          - PointRight: turn right (-angular.z)
          - Fist/None/others: stop

        Only active in NAV; in IDLE publish zeros.
        """
        msg = Twist()

        if self.mode != "NAV":
            return msg  # zeros

        if gesture == "OpenPalm":
            msg.linear.x = +self.lin_speed
        elif gesture == "ThumbUp":
            msg.linear.x = -self.lin_speed
        elif gesture == "PointLeft":
            msg.angular.z = +self.ang_speed
        elif gesture == "PointRight":
            msg.angular.z = -self.ang_speed
        else:
            pass  # stop
        return msg

    def step_once(self) -> bool:
        ok, frame = self.cap.read()
        if not ok:
            return False

        if self.mirror:
            frame = cv2.flip(frame, 1)

        # ====== EC720 inference flow ======
        label_rule = "None"
        label_model = None
        model_prob = None

        res = self.det.process(frame)
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            self.det.draw(frame, hand)
            pix, _ = self.det.to_xy_array(hand, frame.shape)
            feats = pack_features(pix)  # flipped=True by default in your features.py

            # Rule-based (prev_label = current stable, EC720-aligned)
            label_rule = classify_gesture(feats, prev_label=self.label_stable)

            # SVM inference (if loaded)
            if self.clf is not None:
                x = feats_to_vector(feats)
                label_model = self.clf.predict(x)[0]
                if self.has_proba:
                    prob = self.clf.predict_proba(x)[0]
                    model_prob = float(np.max(prob))
                else:
                    model_prob = None

        # ====== Fusion (EC720) ======
        label_raw = fuse_labels(label_rule, label_model, model_prob, prob_th=self.svm_th)

        # ====== Voting (EC720) ======
        self.history.append(label_raw)
        label_vote = Counter(self.history).most_common(1)[0][0]

        # ====== Hysteresis locking (EC720) ======
        if self.candidate is None or label_vote != self.candidate:
            self.candidate = label_vote
            self.cand_count = 1
        else:
            self.cand_count += 1

        locked = self.last_locked
        if self.last_locked == "None":
            if self.cand_count >= self.lock_frames:
                locked = self.candidate
        else:
            if self.candidate != self.last_locked and self.cand_count >= self.switch_confirm_frames:
                locked = self.candidate

        # ====== State machine (EC720) ======
        if self.last_locked == "Fist" and locked == "OpenPalm":
            self.mode = "NAV"
        if locked in ("None", "Fist"):
            self.mode = "IDLE"

        self.last_locked = locked
        self.label_stable = locked

        # ====== Publish cmd_vel ======
        msg = self.map_to_twist(self.label_stable)
        self.pub.publish(msg)
        # --- publish gesture state EVERY frame (for Auto-only / Fusion) ---
        msg_g = String(); msg_g.data = self.label_stable
        msg_m = String(); msg_m.data = self.mode
        msg_e = Bool();   msg_e.data = (self.label_stable in ("Fist", "None"))

        self.pub_gesture.publish(msg_g)
        self.pub_mode.publish(msg_m)
        self.pub_estop.publish(msg_e)

        # ====== FPS stats (optional) ======
        self.fps_cnt += 1
        if time.time() - self.fps_t0 >= 1.0:
            self.fps_disp = self.fps_cnt / (time.time() - self.fps_t0)
            self.fps_t0 = time.time()
            self.fps_cnt = 0

        # ====== HUD ======
        if self.show_camera:
            cv2.putText(frame, f"FPS: {self.fps_disp:.1f}", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
            cv2.putText(frame, f"Rule: {label_rule}", (12, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 50), 2)
            if self.clf is not None:
                if model_prob is not None:
                    cv2.putText(frame, f"SVM: {label_model} ({model_prob:.2f})", (12, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2)
                else:
                    cv2.putText(frame, f"SVM: {label_model}", (12, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2)
            else:
                cv2.putText(frame, "SVM: None", (12, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

            cv2.putText(frame, f"Gesture: {self.label_stable}", (12, 104),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 120), 2)
            cv2.putText(frame, f"Mode: {self.mode}", (12, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 120), 2)

            cv2.imshow("Gesture -> /cmd_vel  (ESC to exit)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                return False

        return True

    def shutdown(self):
        try:
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    rclpy.init()
    node = GestureCmdVel()
    try:
        while rclpy.ok():
            ok = node.step_once()
            rclpy.spin_once(node, timeout_sec=0.0)
            if not ok:
                break
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
