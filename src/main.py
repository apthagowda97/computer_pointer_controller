import time
import os
import cv2
import numpy as np
import logging

from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetection
from landmark_regression import LandmarkRegression
from head_pose import HeadPose
from gaze_estimation import GazeEstimation

from argparse import ArgumentParser


DIR_PATH = os.path.split(os.getcwd())[0]
video_path = os.path.join(DIR_PATH,"bin\\demo.mp4")
face_model_path = os.path.join(DIR_PATH,os.path.join('model\\intel\\face-detection-adas-binary-0001\\FP32-INT1\\face-detection-adas-binary-0001'))
landmark_model_path = os.path.join(DIR_PATH,os.path.join('model\\intel\\landmarks-regression-retail-0009\\FP32-INT8\\landmarks-regression-retail-0009'))
headpose_model_path = os.path.join(DIR_PATH,os.path.join('model\\intel\\head-pose-estimation-adas-0001\\FP32-INT8\\head-pose-estimation-adas-0001'))
gaze_model_path = os.path.join(DIR_PATH,os.path.join('model\\intel\\gaze-estimation-adas-0002\\FP32-INT8\\gaze-estimation-adas-0002'))


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-f", "--face_model", type=str,default=face_model_path,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-he", "--head_model", type=str,default=headpose_model_path,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-l", "--landmark_model", type=str,default=landmark_model_path,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-g", "--gaze_model", type=str,default=gaze_model_path,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-t", "--input_type", type=str,default="video",
                        help="'video', 'cam' or 'image' input type")
    parser.add_argument("-i", "--input_file",type=str,default=video_path,
                        help="Path to image or video file")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on")
    return parser

def inference(args):
    
    time_sheet = {'face_infr':[],'landmark_infr':[],'head_infr':[],'gaze_infr':[],'infr_per_frame':[]}
    
    logging.basicConfig(filename='result.log',level=logging.INFO)
    logging.info("=================================================================================")
    logging.info("Precision(face,landmark,head,gaze): FP32-INT1,FP{0},FP{1},FP{2}".format(\
            args.landmark_model.split("FP")[1].split("\\")[0],
            args.head_model.split("FP")[1].split("\\")[0],
            args.gaze_model.split("FP")[1].split("\\")[0]))

    model_load_start = time.time()

    face_detection= FaceDetection(args.face_model)
    face_detection.load_model()
    landmark_regression= LandmarkRegression(args.landmark_model)
    landmark_regression.load_model()
    head_pose = HeadPose(args.head_model)
    head_pose.load_model()
    gaze_estimation = GazeEstimation(args.gaze_model)
    gaze_estimation.load_model()

    logging.info("4 models load time: {0:.4f}sec".format(time.time()-model_load_start))
    
    mouse_controller = MouseController('high','fast')

    cv2.namedWindow('preview', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('preview', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    input_feeder = InputFeeder(args.input_type,args.input_file)
    input_feeder.load_data()

    total_infr_start = time.time()

    for image in input_feeder.next_batch():
        if image is None:
            break
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
        time_sheet['infr_per_frame'].append(time.time()-face_infr_start)
        cv2.imshow('preview',image)
        mouse_controller.move(x,y)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    logging.info("Face model avg inference per frame: {0:.4f}sec".format(np.mean(time_sheet['face_infr'])))
    logging.info("Landmark model avg inference per frame: {0:.4f}sec".format(np.mean(time_sheet['landmark_infr'])))
    logging.info("Head model avg inference per frame: {0:.4f}sec".format(np.mean(time_sheet['head_infr'])))
    logging.info("Gaze model avg inference per frame: {0:.4f}sec".format(np.mean(time_sheet['gaze_infr'])))
    logging.info("4 Model avg inference per frame: {0:.4f}sec".format(np.mean(time_sheet['infr_per_frame'])))
    logging.info("Total inference time: {0:.4f}sec".format(time.time()-total_infr_start))
    logging.info("====================================END==========================================\n")

    input_feeder.close()
    cv2.destroyAllWindows()

def main():
    args = build_argparser().parse_args()
    inference(args)

if __name__ == '__main__':
    main()


