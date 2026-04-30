# infer_use_svm.py
import argparse
import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np

from hands import HandDetector
from features import pack_features
from rules import classify_gesture

# ========== Tunable parameters ==========
SMOOTH_K = 5                 # Majority vote window size
LOCK_FRAMES = 3              # Lock when same gesture accumulates for >= 3 frames
SWITCH_CONFIRM_FRAMES = 2    # Require >= 2 consecutive frames to confirm switching to a "new" gesture


FEATURE_ORDER = [
    "palm_size",
    "curl_index", "curl_middle", "curl_ring", "curl_pinky",
    "thumb_straight", "thumb_lateral",
    "yaw_norm",      # Recommended to use normalized yaw
]

def feats_to_vector(feats: dict) -> np.ndarray:
    """Convert pack_features dict into a vector according to FEATURE_ORDER (missing keys filled with 0)."""
    return np.array([feats.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float32).reshape(1, -1)

def load_svm(model_path: Path):
    """Load SVM model; return None if loading fails."""
    try:
        import joblib
        if model_path.exists():
            clf = joblib.load(model_path)
            # Some pipeline models do not have predict_proba; check for compatibility
            has_proba = hasattr(clf, "predict_proba")
            return clf, has_proba
    except Exception as e:
        print(f"[WARN] Load SVM failed: {e}")
    return None, False

def fuse_labels(label_rule: str, label_model: str, model_prob: float, prob_th: float = 0.6) -> str:
    """
    Fusion strategy:
    - If model exists and its predicted probability >= prob_th, use the model label;
    - Otherwise, fall back to the rule-based label;
    - If no model exists, this is equivalent to using rule-based only.
    """
    if label_model is not None and model_prob is not None and model_prob >= prob_th:
        return label_model
    return label_rule

def draw_hud(frame, info_lines, x=12, y=24, lh=24):
    """Print multiple lines of HUD information in the top-left corner."""
    for line, color in info_lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += lh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="camera index")
    parser.add_argument("--model", type=str, default="models/svm.pkl", help="path to SVM model")
    parser.add_argument("--svm_th", type=float, default=0.60, help="probability threshold for SVM fusion")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")

    det = HandDetector()

    # Load SVM (fallback to rule-only if not found)
    clf, has_proba = load_svm(Path(args.model))
    if clf is None:
        print("[INFO] SVM model not found or failed to load. Running RULE-only mode.")
    else:
        print(f"[INFO] SVM model loaded: {args.model} | has_proba={has_proba}")

    # Data structures for majority voting and hysteresis
    history = deque(maxlen=SMOOTH_K)
    candidate, cand_count = None, 0

    # State machine: only Fist -> OpenPalm enters NAV
    mode = "IDLE"         # "IDLE" / "NAV"
    last_locked = "None"  # Previously locked stable gesture
    label_stable = "None" # Currently locked stable gesture

    fps_t0, fps_cnt = time.time(), 0
    fps_disp = 0.0

    print("Press ESC to exit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # ========== Inference: detection + feature extraction ==========
        label_rule = "None"
        label_model = None
        model_prob = None

        res = det.process(frame)
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            det.draw(frame, hand)
            pix, _ = det.to_xy_array(hand, frame.shape)
            feats = pack_features(pix)

            # Rule-based method
            label_rule = classify_gesture(feats, prev_label=label_stable)

            # SVM inference (if loaded)
            if clf is not None:
                x = feats_to_vector(feats)
                label_model = clf.predict(x)[0]
                if has_proba:
                    prob = clf.predict_proba(x)[0]
                    # Model confidence is defined as the probability of the predicted class
                    model_prob = float(np.max(prob))
                else:
                    model_prob = None

        # ========== Fusion ==========
        label_raw = fuse_labels(label_rule, label_model, model_prob, prob_th=args.svm_th)

        # ========== Multi-frame voting ==========
        history.append(label_raw)
        cnt = Counter(history)
        label_vote = cnt.most_common(1)[0][0]

        # ========== Hysteresis locking ==========
        if candidate is None or label_vote != candidate:
            candidate = label_vote
            cand_count = 1
        else:
            cand_count += 1

        locked = last_locked  # Keep previous by default
        if last_locked == "None":
            if cand_count >= LOCK_FRAMES:
                locked = candidate
        else:
            if candidate != last_locked and cand_count >= SWITCH_CONFIRM_FRAMES:
                locked = candidate

        # ========== State machine ==========
        # Only when the previous stable gesture is Fist and the newly locked gesture is OpenPalm do we enter NAV
        if last_locked == "Fist" and locked == "OpenPalm":
            mode = "NAV"
        # When the locked gesture is None or Fist, we go back to IDLE
        if locked in ("None", "Fist"):
            mode = "IDLE"

        last_locked = locked
        label_stable = locked

        # ========== FPS statistics ==========
        fps_cnt += 1
        if time.time() - fps_t0 >= 1.0:
            fps_disp = fps_cnt / (time.time() - fps_t0)
            fps_t0 = time.time()
            fps_cnt = 0

        # ========== Draw HUD overlay ==========
        hud = []
        hud.append((f"FPS: {fps_disp:.1f}", (50, 255, 50)))
        hud.append((f"Rule: {label_rule}", (200, 200, 50)))
        if clf is not None:
            if model_prob is not None:
                hud.append((f"SVM: {label_model} ({model_prob:.2f})", (50, 200, 255)))
            else:
                hud.append((f"SVM: {label_model}", (50, 200, 255)))
        else:
            hud.append(("SVM: None", (80, 80, 80)))
        hud.append((f"Gesture: {label_stable}", (0, 220, 120)))
        hud.append((f"Mode: {mode}", (0, 220, 120)))

        draw_hud(frame, hud, x=12, y=26, lh=26)

        cv2.imshow("Gesture Recognition (SVM + Rules)  -  ESC to exit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
