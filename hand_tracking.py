import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

my_hands = mp.solutions.hands.Hands()

try:
    while True:
        _, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = my_hands.process(frame_rgb)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape
                for i, lm in enumerate(hand_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    if i == 4 or i == 8 or i == 12 or i == 16 or i == 20:
                        cv2.circle(frame, (x, y), 13, (255,0,0), -1)
                    # print(i, ": ", x, "\t", y)
                mp.solutions.drawing_utils.draw_landmarks(frame,
                                                      hand_landmarks,
                                                      mp.solutions.hands.HAND_CONNECTIONS)
        
        cv2.imshow("Webcam", frame)
        cv2.waitKey(1)

except KeyboardInterrupt as e:
    pass
except Exception as e:
    print(e)
