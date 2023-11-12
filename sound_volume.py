import cv2
import mediapipe
from hand_tracking_class import Hand_Track
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    indexes = [4, 8, 12, 16, 20]

    img = cv2.imread("D:/Python/Machine Vision/volume.png")
    img = cv2.resize(img, (40, 40))
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    hand_track = Hand_Track(min_detection_confidence=0.55)

    # https://github.com/AndreMiras/pycaw
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol_range = volume.GetVolumeRange()  # min_vol = vol_range[0], max_vol = vol_range[1]
    volume.SetMasterVolumeLevel(vol_range[1], None) # -> to set volume to sth
    vol_std = 100
    ############################################
    
    while True:
        _, frame = cap.read()
        hand_track.find_hands(frame)
        landmarks = hand_track.position(frame, all_fingers_bold=False)
        frame[400:440, 50:90] = img
        
        if len(landmarks):
            p1 = np.array(landmarks[indexes[0]][2:4])
            p2 = np.array(landmarks[indexes[1]][2:4])
            pc = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))

            dist = np.linalg.norm(p1 - p2)
            # print(dist) min=8, max=125

            volume_to_set = np.interp(dist, [12, 108], [vol_range[0], vol_range[1]])
            volume.SetMasterVolumeLevel(volume_to_set, None)

            vol_std = np.interp(dist, [13, 108], [0, 100])

            cv2.line(frame, p1, p2, (255,0,0), 3)
            if dist < 12:
                cv2.circle(frame, pc, 7, (0,255,255), -1)
            elif dist > 109:
                cv2.circle(frame, pc, 7, (0,255,255), -1)
            else:
                cv2.circle(frame, pc, 7, (255,0,0), -1)

            if vol_std > 10:
                for i in range(int(vol_std/10) + 1):
                    cv2.rectangle(frame, (95 + 10*i, 400), (100 + 10*i, 440), (0,255,0), -1)
            
        cv2.putText(frame, f"{int(vol_std)} %", (50, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)
        cv2.imshow("volume changer", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        print(e)
        
