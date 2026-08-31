import cv2
import mediapipe as mp

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# Finger tip landmark IDs (Thumb, Index, Middle, Ring, Pinky)
FINGER_TIPS = [4, 8, 12, 16, 20]
# Finger MCP (knuckle) IDs for comparison
FINGER_MCPS = [2, 6, 10, 14, 18]


def count_fingers(hand_landmarks, handedness_label):
    """
    Returns number of fingers raised for one hand.
    handedness_label: 'Left' or 'Right' (MediaPipe's label)
    """
    lm = hand_landmarks.landmark
    fingers_up = 0

    # ── Thumb ────────────────────────────────────────────────────────────────
    # Thumb logic differs: compare x-axis because thumb moves sideways
    # For Right hand: tip.x < mcp.x means thumb is open (mirrored camera)
    # For Left  hand: tip.x > mcp.x means thumb is open
    tip_x = lm[FINGER_TIPS[0]].x
    mcp_x = lm[FINGER_MCPS[0]].x
    if handedness_label == "Right":
        if tip_x < mcp_x:
            fingers_up += 1
    else:
        if tip_x > mcp_x:
            fingers_up += 1

    # ── Other 4 fingers ───────────────────────────────────────────────────────
    # Finger is raised if tip.y < pip.y  (y increases downward in image)
    PIP_IDS = [6, 10, 14, 18]   # Proximal Inter-Phalangeal joints
    for tip_id, pip_id in zip(FINGER_TIPS[1:], PIP_IDS):
        if lm[tip_id].y < lm[pip_id].y:
            fingers_up += 1

    return fingers_up


def draw_skeleton(frame, hand_landmarks):
    """Draws hand landmark points and connection lines."""
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=5),   # joints
        mp_drawing.DrawingSpec(color=(255, 180, 0),  thickness=3, circle_radius=2),  # bones
    )


def put_outlined_text(frame, text, pos, font_scale=1.2, color=(255, 255, 255),
                      thickness=2, outline_color=(0, 0, 0)):
    """Draws text with a dark outline for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    # Outline
    cv2.putText(frame, text, (x - 2, y - 2), font, font_scale, outline_color, thickness + 3)
    cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, outline_color, thickness + 3)
    # Main text
    cv2.putText(frame, text, pos, font, font_scale, color, thickness)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=4,          # detect up to 4 hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        print("[INFO] Hand Finger Detection started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR → RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_finger_counts = []   # list of (label, count, cx, cy)

            if results.multi_hand_landmarks:
                for idx, (hand_lm, hand_info) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # ── Draw skeleton ─────────────────────────────────────
                    draw_skeleton(frame, hand_lm)

                    # ── Finger count ──────────────────────────────────────
                    label = hand_info.classification[0].label  # 'Left' / 'Right'
                    count = count_fingers(hand_lm, label)

                    # Wrist landmark as anchor for label position
                    wrist = hand_lm.landmark[0]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)

                    hand_finger_counts.append((label, count, cx, cy))

                    # Per-hand label
                    color = (0, 220, 255) if label == "Right" else (255, 180, 0)
                    put_outlined_text(frame, f"{label}: {count}", (cx - 30, cy + 50),
                                      font_scale=0.9, color=color)

            # ── Deduction logic ───────────────────────────────────────────────
            # Sort by x-position (leftmost hand = index 0 in mirrored view)
            # Main hand = first detected; extra hands are deducted
           
            num_hands = len(hand_finger_counts)

            if num_hands == 0:
                net_count = 0
                net_text = "No Hand Detected"
                net_color = (180, 180, 180)

            else:
                  # Add fingers from all detected hands
                net_count = sum(count for _, count, _, _ in hand_finger_counts)

                net_text = f"Total Fingers: {net_count}"
                net_color = (0, 255, 128)
            # ── Finger counting logic ─────────────────────────────────────

            num_hands = len(hand_finger_counts)

            if num_hands == 0:

                 net_count = 0
                 net_text = "No Hand Detected"
                 net_color = (180, 180, 180)

            else:

                 # Count fingers from all detected hands
                net_count = sum(
                     count for _, count, _, _ in hand_finger_counts
                 )

                 
            # ── HUD overlay ───────────────────────────────────────────────────
            # Dark banner at top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            put_outlined_text(frame, net_text, (20, 60),
                              font_scale=1.4, color=net_color, thickness=2)

            # Bottom hint
            cv2.putText(frame, "Press 'q' to quit",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (160, 160, 160), 1)

            # Hand count badge
            badge_text = f"Hands: {num_hands}"
            put_outlined_text(frame, badge_text, (w - 180, 60),
                              font_scale=1.0, color=(200, 200, 255))

            cv2.imshow("Hand Finger Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()

