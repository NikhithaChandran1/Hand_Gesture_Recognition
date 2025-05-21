import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks indices
tipIds = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tipIds[0]].x < hand_landmarks.landmark[tipIds[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tipIds[id]].y < hand_landmarks.landmark[tipIds[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for better speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pTime = 0

# Cooldown for gestures (in seconds)
cooldown = 0.3
last_gesture_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to act as a mirror
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(hand_landmarks)

            # Gesture logic with cooldown
            if fingers == [0, 1, 0, 0, 0]:  # Swipe right
                if current_time - last_gesture_time > cooldown:
                    pyautogui.press('right')
                    last_gesture_time = current_time
                cv2.putText(frame, 'Swipe Right', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

            elif fingers == [0, 1, 1, 0, 0]:  # Swipe left
                if current_time - last_gesture_time > cooldown:
                    pyautogui.press('left')
                    last_gesture_time = current_time
                cv2.putText(frame, 'Swipe Left', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

            elif fingers == [1, 1, 1, 1, 1]:  # Jump
                if current_time - last_gesture_time > cooldown:
                    pyautogui.press('up')
                    last_gesture_time = current_time
                cv2.putText(frame, 'Jump', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

            elif fingers == [0, 0, 0, 0, 0]:  # Roll (hoverboard)
                if current_time - last_gesture_time > cooldown:
                    pyautogui.press('down')
                    last_gesture_time = current_time
                cv2.putText(frame, 'Roll (Hoverboard)', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime)!=0 else 0
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Subway Surfers Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

 
