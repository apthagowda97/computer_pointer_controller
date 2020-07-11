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

from argparse import ArgumentParser

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-f", "--face_model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-he", "--head_model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-l", "--landmark_model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-g", "--gaze_model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-t", "--input_type", required=True, type=str,default="cam",
                        help="'video', 'cam' or 'image' input type")
    parser.add_argument("-i", "--input_file",type=str,default=None,
                        help="Path to image or video file")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on")
    return parser

def inference(args):
    
    time_sheet = {'model_load':[],'face_infr':[],'landmark_infr':[],'head_infr':[],'gaze_infr':[]}

    model_load_start = time.time()

    face_detection= FaceDetection(args.face_model)
    face_detection.load_model()
    landmark_regression= LandmarkRegression(args.landmark_model)
    landmark_regression.load_model()
    head_pose = HeadPose(args.head_model)
    head_pose.load_model()
    gaze_estimation = GazeEstimation(args.gaze_model)
    gaze_estimation.load_model()

    time_sheet['model_load'].append(time.time()-model_load_start)

    mouse_controller = MouseController('medium','fast')

    cv2.namedWindow('preview', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('preview', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    input_feeder = InputFeeder(args.input_type,args.input_file)
    input_feeder.load_data()

    for image in input_feeder.next_batch():

        face_infr_start = time.time()
        face_image = face_detection.predict(image)
        time_sheet['face_infr'].append(time.time()-face_infr_start)

        landmark_infr_start = time.time()
        left_eye_image,right_eye_image = landmark_regression.predict(np.copy(face_image))
        time_sheet['landmark_infr'].append(time.time()-landmark_infr_start)

        head_infr_start = time.time()       
        head_pose_angles = head_pose.predict(np.copy(face_image))
        time_sheet['head_infr'].append(time.time()-head_infr_start)

        gaze_infr_start = time.time()
        x,y,z = gaze_estimation.predict(left_eye_image,right_eye_image,head_pose_angles)
        time_sheet['gaze_infr'].append(time.time()-gaze_infr_start)

        cv2.imshow('preview',image)
        mouse_controller.move(x,y)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            return time_sheet

    input_feeder.close()
    return time_sheet

def main():
    args = build_argparser().parse_args()
    time_sheet = inference(args)
    print("Model load time: ",np.mean(time_sheet['model_load']))
    print("Face infr time: ",np.mean(time_sheet['face_infr']))
    print("landmark infr time: ",np.mean(time_sheet['landmark_infr']))
    print("head infr time: ",np.mean(time_sheet['head_infr']))
    print("gaze infr time: ",np.mean(time_sheet['gaze_infr']))



if __name__ == '__main__':
    main()


