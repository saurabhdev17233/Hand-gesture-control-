import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start Webcam
cap = cv2.VideoCapture(0)

prev_x = 0
cooldown = 1
last_action = time.time()

# Finger tip landmarks
tips_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if lm_list:
                # Hand center
                cx = (lm_list[0][0] + lm_list[9][0]) // 2
                cy = (lm_list[0][1] + lm_list[9][1]) // 2
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                # Movement detection
                dx = cx - prev_x
                now = time.time()

                if now - last_action > cooldown:
                    if dx > 40:
                        print("RIGHT")
                        pyautogui.press("right")
                        last_action = now
                    elif dx < -40:
                        print("LEFT")
                        pyautogui.press("left")
                        last_action = now
                    prev_x = cx

                # Finger detection
                fingers = []

                # Thumb (left vs right hand handled by x coordinate)
                if lm_list[tips_ids[0]][0] > lm_list[tips_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other fingers
                for i in range(1, 5):
                    if lm_list[tips_ids[i]][1] < lm_list[tips_ids[i] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = sum(fingers)

                if now - last_action > cooldown:
                    if total_fingers >= 4:
                        print("JUMP (OPEN HAND)")
                        pyautogui.press("up")
                        last_action = now
                    elif total_fingers <= 1:
                        print("SLIDE (CLOSED FIST)")
                        pyautogui.press("down")
                        last_action = now

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Temple Run Hand Control", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
