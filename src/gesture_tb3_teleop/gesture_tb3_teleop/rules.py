#rules.py
def classify_gesture(feats, prev_label=None):
    c_idx = feats["curl_index"]
    c_mid = feats["curl_middle"]
    c_rng = feats["curl_ring"]
    c_pky = feats["curl_pinky"]
    thumb_s = feats["thumb_straight"]
    yaw    = feats["yaw"]
    ps     = feats["palm_size"] + 1e-6
    idx_dx = feats["index_dir_x"]
    tipsum = feats["tipdist_sum"] / ps  # Normalized distance (relative to palm scale)
    thumb_upy = feats.get("thumb_tip_up", 0.0)

    # Thresholds
    STRAIGHT = 0.90     # Higher = straighter
    CURLED   = 0.70     # Lower = more curled (slightly higher to avoid being too strict)
    TIP_NEAR = 4.1      # Fist: threshold of total fingertip distance / palm size (smaller = closer)
    TIP_FAR  = 5.5      # OpenPalm: total fingertip distance is larger
    IDX_DIR  = 3.0      # X-component threshold for index finger direction (pixels / relative scale, tune this!)
    THUMB_UP_Y = 1.0    # Thumb "upward" strength (optional; if missing, default 0.0, does not change original logic)

    # OpenPalm: four fingers straight & fingertips far
    if (c_idx > STRAIGHT and c_mid > STRAIGHT and c_rng > STRAIGHT and c_pky > STRAIGHT) and (tipsum > TIP_FAR):
        return "OpenPalm"

    # Fist: five fingers curled + fingertips close to palm
    if (thumb_s < CURLED and c_idx < CURLED and c_mid < CURLED and c_rng < CURLED and c_pky < CURLED and tipsum < TIP_NEAR):
        return "Fist"

    # ThumbUp: thumb straight + thumb up + other fingers mostly curled
    if (thumb_s > STRAIGHT and thumb_upy > THUMB_UP_Y and c_idx < CURLED and c_mid < CURLED and c_rng < CURLED and c_pky < CURLED):
        return "ThumbUp"

    # PointRight: index straight + others curled + index clearly pointing right
    if (c_idx > STRAIGHT and c_mid < CURLED and c_rng < CURLED and c_pky < CURLED) and (idx_dx < -IDX_DIR):
        return "PointRight"

    # PointLeft: similarly pointing left
    if (c_idx > STRAIGHT and c_mid < CURLED and c_rng < CURLED and c_pky < CURLED) and (idx_dx > IDX_DIR):
        return "PointLeft"

    return prev_label if prev_label is not None else "None"
