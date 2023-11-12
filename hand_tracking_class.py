import cv2
import mediapipe as mp

class Hand_Track:
    def __init__(self, static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5):
        self.my_hands = mp.solutions.hands.Hands(static_image_mode=static_image_mode,
                                                 max_num_hands=max_num_hands,
                                                 min_detection_confidence=min_detection_confidence)

    def find_hands(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.my_hands.process(frame_rgb)
        if self.results.multi_hand_landmarks:
            if draw:
                for hl in self.results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)

    def position(self, frame, draw=True, all_fingers_bold=True):
        landmarks = []
        indexes = [4, 8, 12, 16, 20]
        if self.results.multi_hand_landmarks:
            for handNo, hand_landmark in enumerate(self.results.multi_hand_landmarks):
                h, w, _ = frame.shape
                for i, lm in enumerate(hand_landmark.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    if draw and i in indexes and all_fingers_bold:
                        cv2.circle(frame, (x, y), 8, (255,0,0), -1)
                    elif draw and i in indexes[:2] and not all_fingers_bold:
                        cv2.circle(frame, (x, y), 8, (255,0,0), -1)
                    landmarks.append([handNo, i, x, y])
        return landmarks


def main():
    cap = cv2.VideoCapture(0)
    hand_track = Hand_Track()
    while True:
        _, frame = cap.read()
        hand_track.find_hands(frame)
        landmarks = hand_track.position(frame)
        cv2.imshow("webcam", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        #print(len(landmarks)
        pass
    except Exception as e:
        print(e)
