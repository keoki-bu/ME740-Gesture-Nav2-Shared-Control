#features.py
import numpy as np

# MediaPipe Hands keypoint indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

def _vec(a, b):
    return b - a

def _norm(v, eps=1e-6):
    n = np.linalg.norm(v)
    return v / (n + eps), n

def palm_size(pix):
    """Measure palm scale as the distance from WRIST to MIDDLE_MCP."""
    v = pix[MIDDLE_MCP] - pix[WRIST]
    return np.linalg.norm(v)

def finger_curl_score(pix, finger):
    """
    Measure finger curl using angles between adjacent joints:
    larger = straighter, smaller = more bent.
    Return a combined curl score (0–1) based on [pip_angle, dip_angle, tip_dir].
    """
    mcp, pip, dip, tip = finger
    v1, _ = _norm(_vec(pix[mcp], pix[pip]))
    v2, _ = _norm(_vec(pix[pip], pix[dip]))
    v3, _ = _norm(_vec(pix[dip], pix[tip]))
    # Cosine of angle (-1~1), mapped to 0~1 (1 = fully straight)
    def straightness(a, b):
        cosang = np.clip(np.dot(a, b), -1.0, 1.0)
        ang = np.arccos(cosang)  # 0 = straight line, pi = opposite direction
        return 1.0 - (ang / np.pi)  # straight ≈ 1, 90° bend ≈ 0.5, folded ≈ 0
    s1 = straightness(v1, v2)
    s2 = straightness(v2, v3)
    # Fingertip direction relative to palm center (more outward = straighter)
    palm = pix[WRIST]
    tip_dir_vec, _ = _norm(_vec(palm, pix[tip]))
    palm_dir_vec, _ = _norm(_vec(palm, pix[mcp]))
    s3 = (np.dot(tip_dir_vec, palm_dir_vec) + 1) / 2  # -1~1 → 0~1
    return float(np.mean([s1, s2, s3]))

def thumb_direction_score(pix):
    """Determine whether the thumb is straight and its direction (radial relative to the palm)."""
    v_cmc_mcp, _ = _norm(_vec(pix[THUMB_CMC], pix[THUMB_MCP]))
    v_mcp_tip, _ = _norm(_vec(pix[THUMB_MCP], pix[THUMB_TIP]))
    # Degree of straightness
    straight = (np.dot(v_cmc_mcp, v_mcp_tip) + 1) / 2
    # Lateral direction (>0 may indicate “outward”)
    lateral = v_mcp_tip[0]
    return straight, lateral  # (0~1, negative = left / positive = right)

def hand_yaw(pix):
    """
    Rough estimate of hand “yaw”: x-direction from index_mcp to pinky_mcp.
    >0 means facing right, <0 facing left.
    """
    v = pix[PINKY_MCP] - pix[INDEX_MCP]
    return v[0]

def fingertip_sum_dist_to_palm(pix):
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    palm = pix[WRIST]
    d = 0.0
    for t in tips:
        d += np.linalg.norm(pix[t] - palm)
    return d

def index_direction_x(pix, flipped=True):
    """X component of index finger direction (tip - mcp); reversed after horizontal flip."""
    v = pix[INDEX_TIP] - pix[INDEX_MCP]
    x = v[0]
    return -x if flipped else x

def pack_features(pix, flipped=True):
    ps = palm_size(pix)
    curls = {name: finger_curl_score(pix, FINGERS[name]) for name in FINGERS}
    thumb_straight, thumb_lat = thumb_direction_score(pix)
    yaw = hand_yaw(pix)
    idx_dir_x = index_direction_x(pix, flipped=flipped)
    tipdist_sum = fingertip_sum_dist_to_palm(pix)
    # Compute how “upward” the thumb is (negative y is up, so use wrist_y - thumb_tip_y)
    wrist_y = pix[WRIST][1]
    thumb_tip_y = pix[THUMB_TIP][1]
    up_amount = (wrist_y - thumb_tip_y) / (palm_size(pix) + 1e-6)  # Normalize by palm scale
    # Compute hand bounding-box size
    min_xy = np.min(pix, axis=0)
    max_xy = np.max(pix, axis=0)
    bbox_w, bbox_h = max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]
    bbox_size = float(max(bbox_w, bbox_h) + 1e-6)
    yaw_norm = float(yaw / bbox_size)

    feats = {
        "palm_size": ps,
        "curl_index": curls["index"],
        "curl_middle": curls["middle"],
        "curl_ring": curls["ring"],
        "curl_pinky": curls["pinky"],
        "thumb_straight": thumb_straight,
        "thumb_lateral": thumb_lat,
        "yaw": yaw,
        "index_dir_x": idx_dir_x,
        "tipdist_sum": tipdist_sum,
        "thumb_tip_up": float(up_amount),
        "bbox_size": bbox_size,
        "yaw_norm": yaw_norm,
    }
    return feats
