import cv2
import mediapipe
import os
from hand_tracking_class import Hand_Track
import numpy as np


def finger_counter(landmarks, indexes):
    fingers = []
    # thumb
    if landmarks[indexes[0]][2] < landmarks[indexes[0]-1][2]:
        fingers.append(1)
    else:
        fingers.append(0)
    # 4 fingers
    for i in indexes[1:]:
        if landmarks[i][3] < landmarks[i-2][3]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def main():
    path = "C:/pouria/Python/Machine Vision/paint/toolbar"
    toolbars = os.listdir(path)
    toolbar = []
    for toolpath in toolbars[:-2]:
        img = cv2.imread(path + '/' + toolpath)
        img = cv2.resize(img, (320, 32))
        toolbar.append(img)
    for toolpath in toolbars[-2:]:
        img = cv2.imread(path + '/' + toolpath)
        img = cv2.resize(img, (50, 100))
        toolbar.append(img)
    
    indexes = [4, 8, 12, 16, 20]
    menu = toolbar[0]
    menu_eraser = toolbar[-1]
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    hand_track = Hand_Track(min_detection_confidence=0.6, max_num_hands=1)

    x0, y0 = 0, 0
    color = (255,0,0)
    thikness = 8

    blank = np.zeros((480, 640, 3), np.uint8)
    
    while True:
        _, frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        
        frame[:32, :320] = menu
        frame[:100, 590:640] = menu_eraser
        
        hand_track.find_hands(frame)
        landmarks = hand_track.position(frame, all_fingers_bold=False)

        if len(landmarks):
            x1, y1 = int(landmarks[indexes[1]][2]), int(landmarks[indexes[1]][3])
            x2, y2 = int(landmarks[indexes[2]][2]), int(landmarks[indexes[2]][3])
            #select
            if y1<landmarks[indexes[1]-2][3] and y2<landmarks[indexes[2]-2][3]:
                cv2.circle(frame, (x2,y2), 8, (255,0,0), -1)
                
                if 64<x2<128 and 0<y2<45:
                    color = (255,0,0)
                    menu = toolbar[0]
                elif 128<x2<192 and 0<y2<45:
                    color = (255,0,255)
                    menu = toolbar[1]
                elif 192<x2<256 and 0<y2<45:
                    color = (0,255,0)
                    menu = toolbar[2]
                elif 256<x2<320 and 0<y2<45:
                    color = (0,0,0)
                    menu = toolbar[3]
                elif 590<x2<640 and 0<y2<50:
                    thikness = 8
                    menu_eraser = toolbar[-1]
                elif 590<x2<640 and 50<y2<105:
                    thikness = 20
                    menu_eraser = toolbar[-2]
            #draw
            elif y1<landmarks[indexes[1]-2][3]:
                if x0==0 and y0==0:
                    x0, y0 = x1, y1

                if color == (0,0,0):
                    cv2.circle(frame, (x1, y1), thikness, color, -1)
                    cv2.line(blank, (x0,y0), (x1,y1), color, thikness)
                else:
                    cv2.circle(frame, (x1, y1), thikness, color, -1)
                    cv2.line(blank, (x0,y0), (x1,y1), color, thikness)
                x0, y0 = x1, y1

            fingers = finger_counter(landmarks, indexes)

            if all(x==1 for x in fingers):
                blank = np.zeros((480, 640, 3), np.uint8)

        gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, blank)

        cv2.imshow("paint", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        print(e)
        
