import cv2
import mediapipe
from hand_tracking_class import Hand_Track

def main():
    indexes = [4, 8, 12, 16, 20]
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 600)
    cap.set(4, 600)
    hand_track = Hand_Track(min_detection_confidence=0.55)
    while True:
        fingers = []
        _, frame = cap.read()
        hand_track.find_hands(frame)
        landmarks = hand_track.position(frame)
        if len(landmarks):
            # thumb
            if landmarks[indexes[0]][2] > landmarks[indexes[0]-1][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 fingers
            for i in indexes[1:]:
                if landmarks[i][3] < landmarks[i-2][3]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            count = sum(fingers)
            cv2.putText(frame, str(count), (50, 100), cv2.FONT_HERSHEY_PLAIN, 8, (0,255,0), 7)
            #print(count, end=' ')
                
        cv2.imshow("finger counter", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        print(e)
