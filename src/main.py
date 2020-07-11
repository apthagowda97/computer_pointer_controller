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
    
    time_sheet = {'model_load':[],'face_infr':[],'landmark_infr':[],'head_infr':[],'gaze_infr':[],'total_infr':0}
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

    mouse_controller = MouseController('high','fast')

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
            time_sheet['total_infr'] = time.time()-model_load_start
            return time_sheet

    time_sheet['total_infr'] = time.time()-model_load_start
    input_feeder.close()
    return time_sheet

def end_result(args,time_sheet):

    file = open("result.txt", "a")

    avg_face = np.mean(time_sheet['face_infr'])
    avg_landmark = np.mean(time_sheet['face_infr'])
    avg_head = np.mean(time_sheet['head_infr'])
    avg_gaze = np.mean(time_sheet['gaze_infr'])

    data = "\n\
            Precision(face,landmark,head,gaze): FP32-INT1,FP{0},FP{1},FP{2}\n\
            4 models load time: {3:.3f}sec\n\
            Face model avg inference per frame: {4:.3f}sec\n\
            Landmark model avg inference per frame: {5:.3f}sec\n\
            Head model avg inference per frame: {6:.3f}sec\n\
            Gaze model avg inference per frame: {7:.3f}sec\n\
            4 Model avg inference per frame: {8:.3f}sec\n\
            Total inference time: {9:.3f}sec\n\
            ----------------------------------------\
            ".format(\
                args.landmark_model.split("FP")[1].split("\\")[0],
                args.head_model.split("FP")[1].split("\\")[0],
                args.gaze_model.split("FP")[1].split("\\")[0],
                np.mean(time_sheet['model_load']),avg_face,avg_landmark,avg_head,avg_gaze,\
                np.mean(avg_face+avg_landmark+avg_head+avg_gaze),time_sheet['total_infr'])
    file.write(data)
    file.close()

def main():
    args = build_argparser().parse_args()
    time_sheet = inference(args)
    end_result(args,time_sheet)

if __name__ == '__main__':
    main()


