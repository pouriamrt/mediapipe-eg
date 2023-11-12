import cv2
import mediapipe as mp
from hand_tracking_class import Hand_Track

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 600)
    cap.set(4, 600)
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
        pass
    except Exception as e:
        print(e)
