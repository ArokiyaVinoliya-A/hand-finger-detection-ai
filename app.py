import cv2
import mediapipe as mp
import gradio as gr


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCPS = [2, 6, 10, 14, 18]
PIP_IDS = [6, 10, 14, 18]


def count_fingers(hand_landmarks, handedness_label):

    lm = hand_landmarks.landmark
    fingers_up = 0

    # Thumb
    tip_x = lm[FINGER_TIPS[0]].x
    mcp_x = lm[FINGER_MCPS[0]].x

    if handedness_label == "Right":
        if tip_x < mcp_x:
            fingers_up += 1
    else:
        if tip_x > mcp_x:
            fingers_up += 1

    # Other four fingers
    for tip_id, pip_id in zip(FINGER_TIPS[1:], PIP_IDS):

        if lm[tip_id].y < lm[pip_id].y:
            fingers_up += 1

    return fingers_up


def process_frame(frame):

    if frame is None:
        return None

    # Gradio gives RGB → convert to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w, _ = frame.shape

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_finger_counts = []

        if results.multi_hand_landmarks:

            for hand_lm, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 128),
                        thickness=2,
                        circle_radius=5
                    ),
                    mp_drawing.DrawingSpec(
                        color=(255, 180, 0),
                        thickness=3,
                        circle_radius=2
                    )
                )

                label = hand_info.classification[0].label

                count = count_fingers(
                    hand_lm,
                    label
                )

                hand_finger_counts.append(count)

                wrist = hand_lm.landmark[0]

                x = int(wrist.x * w)
                y = int(wrist.y * h)

                cv2.putText(
                    frame,
                    f"{label}: {count}",
                    (x - 30, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 220, 255),
                    2
                )

        # Total fingers
        num_hands = len(hand_finger_counts)

        if num_hands == 0:

            total_fingers = 0
            text = "No Hand Detected"

        else:

            total_fingers = sum(hand_finger_counts)

            text = f"Total Fingers: {total_fingers}"

        # Top banner
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 90),
            (20, 20, 20),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.55,
            frame,
            0.45,
            0,
            frame
        )

        cv2.putText(
            frame,
            text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 128),
            3
        )

        cv2.putText(
            frame,
            f"Hands: {num_hands}",
            (w - 180, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (200, 200, 255),
            2
        )

    # BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(
        sources=["webcam"],
        type="numpy",
        streaming=True
    ),
    outputs=gr.Image(),
    title="🤚 Hand Finger Detection & Skeleton Lines",
    description=(
        "Real-time hand detection using "
        "Python, OpenCV, MediaPipe and AI."
    ),
    live=True
)


if __name__ == "__main__":
    demo.launch()