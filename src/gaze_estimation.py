from openvino.inference_engine import IECore,IENetwork
import cv2
import numpy as np

class GazeEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions=extensions
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=list(self.model.inputs.keys())
        self.input_shape = self.model.inputs['left_eye_image'].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        
    def load_model(self):
        self.net = IECore().load_network(network = self.model, device_name = self.device,num_requests=1)

    def predict(self, left_eye_image,right_eye_image,head_pose_angles):
        self.preprocess_left_eye_image  = self.preprocess_input(np.copy(left_eye_image))
        self.preprocess_right_eye_image  = self.preprocess_input(np.copy(right_eye_image))
        self.head_pose_angles = head_pose_angles
        self.net.start_async(request_id=0, inputs={'head_pose_angles': self.head_pose_angles,\
                                                 'left_eye_image': self.preprocess_left_eye_image,\
                                                  'right_eye_image': self.preprocess_right_eye_image})
        if self.net.requests[0].wait(-1) == 0:
            outputs = self.net.requests[0].outputs
            outputs = self.preprocess_output(outputs)
        return outputs

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):
        return outputs[self.output_name].squeeze()