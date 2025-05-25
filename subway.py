import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_draw = mp.solutions.drawing_utils

# Tip landmark IDs
tipIds = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks):
    fingers = []
    # Thumb
    if hand_landmarks.landmark[tipIds[0]].x < hand_landmarks.landmark[tipIds[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tipIds[id]].y < hand_landmarks.landmark[tipIds[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

position_buffer = deque(maxlen=4)
last_action_time = 0
cooldown = 0.25  # seconds
movement_threshold = 35
action_text = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    current_time = time.time()
    hand_present = False
    detected_action = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            hand_present = True
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(hand_landmarks)

            # Get center position of the hand
            cx = int(hand_landmarks.landmark[9].x * w)
            cy = int(hand_landmarks.landmark[9].y * h)
            position_buffer.append((cx, cy))

            # Movement detection when all fingers are up
            if fingers == [1, 1, 1, 1, 1] and len(position_buffer) >= 2:
                dx = position_buffer[-1][0] - position_buffer[0][0]
                dy = position_buffer[-1][1] - position_buffer[0][1]

                if abs(dx) > abs(dy):
                    if dx > movement_threshold:
                        detected_action = "Move Right"
                    elif dx < -movement_threshold:
                        detected_action = "Move Left"
                else:
                    if dy < -movement_threshold:
                        detected_action = "Jump"
                    elif dy > movement_threshold:
                        detected_action = "Slide"

            # All fingers closed → Hoverboard
            elif fingers == [0, 0, 0, 0, 0]:
                detected_action = "Hoverboard"

    # Trigger action immediately (no delay)
    if detected_action and (current_time - last_action_time > cooldown):
        if detected_action == "Move Right":
            pyautogui.press("right")
        elif detected_action == "Move Left":
            pyautogui.press("left")
        elif detected_action == "Jump":
            pyautogui.press("up")
        elif detected_action == "Slide":
            pyautogui.press("down")
        elif detected_action == "Hoverboard":
            pyautogui.press("space")
        action_text = detected_action
        last_action_time = current_time

    # Display current action
    if action_text:
        cv2.putText(frame, f'Action: {action_text}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # FPS
    fps = int(1 / max((time.time() - current_time), 1e-5))
    cv2.putText(frame, f'FPS: {fps}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if not hand_present:
        position_buffer.clear()
        action_text = ""

    cv2.imshow("Subway Surfers Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

