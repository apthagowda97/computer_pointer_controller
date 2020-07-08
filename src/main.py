import time
import os
import cv2
import numpy as np

from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetection
from landmark_regression import LandmarkRegression
from head_pose import HeadPose
from gaze_estimation import GazeEstimation

import matplotlib.pyplot as plt

DIR_PATH = os.path.split(os.getcwd())[0]
video_file = os.path.join(DIR_PATH,"bin\\demo.mp4")

face_model = os.path.join(DIR_PATH,os.path.join('model\\intel\\face-detection-adas-binary-0001\\FP32-INT1\\face-detection-adas-binary-0001'))
landmark_model= os.path.join(DIR_PATH,os.path.join('model\\intel\\landmarks-regression-retail-0009\\FP32-INT8\\landmarks-regression-retail-0009'))
headpose_model= os.path.join(DIR_PATH,os.path.join('model\\intel\\head-pose-estimation-adas-0001\\FP32-INT8\\head-pose-estimation-adas-0001'))
gaze_model = os.path.join(DIR_PATH,os.path.join('model\\intel\\gaze-estimation-adas-0002\\FP32-INT8\\gaze-estimation-adas-0002'))

face_detection= FaceDetection(face_model)
face_detection.load_model()

landmark_regression= LandmarkRegression(landmark_model)
landmark_regression.load_model()

head_pose = HeadPose(headpose_model)
head_pose.load_model()

gaze_estimation = GazeEstimation(gaze_model)
gaze_estimation.load_model()

mouse_controller = MouseController('medium','fast')

input_feeder = InputFeeder('cam',video_file)
input_feeder.load_data()

cv2.namedWindow('preview', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('preview', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

start_time = time.time()
for i in range(20):
    image = next(iter(input_feeder.next_batch()))
    face_image = face_detection.predict(image)
    left_eye_image,right_eye_image = landmark_regression.predict(np.copy(face_image))
    head_pose_angles = head_pose.predict(np.copy(face_image))
    x,y,z = gaze_estimation.predict(left_eye_image,right_eye_image,head_pose_angles)
    cv2.imshow('preview',image)
    mouse_controller.move(x,y)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
print("Infr time per frame: ",(time.time()-start_time)/20)
cv2.destroyWindow("preview")
input_feeder.close()