import cv2
import mediapipe as mp
import numpy as np


class Pose_Track:
    def __init__(self, static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):

        self.pose_track = mp.solutions.pose.Pose(static_image_mode=False,
                                                 model_complexity=1,
                                                 smooth_landmarks=True,
                                                 enable_segmentation=False,
                                                 smooth_segmentation=True,
                                                 min_detection_confidence=0.5,
                                                 min_tracking_confidence=0.5)

    def find_pose(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.results = self.pose_track.process(frame_rgb)

        if draw:
            if self.results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    def position(self, frame):
        landmarks = []
        h, w, _ = frame.shape
        if self.results.pose_landmarks:
            for i, pose_landmarks in enumerate(self.results.pose_landmarks.landmark):
                x, y = int(pose_landmarks.x * w), int(pose_landmarks.y * h)
                landmarks.append([i, x, y])
        return landmarks


def main():
    cap = cv2.VideoCapture(r'D:\Python\Machine Vision\body landmark\clips\1.mp4')
    cap.set(3, 600)
    cap.set(4, 600)

    pose = Pose_Track()
    
    while True:
        _, frame = cap.read()

        pose.find_pose(frame)

        landmarks = pose.position(frame)
        
        cv2.imshow("pose", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        print(e)
