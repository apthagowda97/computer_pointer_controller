from openvino.inference_engine import IECore,IENetwork
import cv2
import numpy as np

class FaceDetection:
    '''
    Class for the Face Detection Model.
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

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        
    def load_model(self):
        self.net = IECore().load_network(network = self.model, device_name = self.device,num_requests=1)

    def predict(self, image):
        self.preprocess_image  = self.preprocess_input(np.copy(image))
        self.net.start_async(request_id=0, inputs={self.input_name: self.preprocess_image})
        if self.net.requests[0].wait(-1) == 0:
            outputs = self.net.requests[0].outputs
            face_image = self.preprocess_output(outputs,image)
        return face_image

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs,image):
        box = outputs[self.output_name][0][0][0]
        xmin = int(box[3] * image.shape[1])
        ymin = int(box[4] * image.shape[0])
        xmax = int(box[5] * image.shape[1])
        ymax = int(box[6] * image.shape[0])
        return image[ymin:ymax,xmin:xmax]