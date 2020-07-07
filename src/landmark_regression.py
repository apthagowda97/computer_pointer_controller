from openvino.inference_engine import IECore,IENetwork
import cv2
import numpy as np

class LandmarkRegression:
    '''
    Class for the Landmark Regression Model.
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
        self.image = image
        self.preprocess_image  = self.preprocess_input(np.copy(image))
        self.net.start_async(request_id=0, inputs={self.input_name: self.preprocess_image})
        if self.net.requests[0].wait(-1) == 0:
            outputs = self.net.requests[0].outputs
            left_eye,right_eye = self.preprocess_output(outputs,image)
        return left_eye,right_eye 

    def clip(self,x,y,image,factor=30):
        ymin = np.clip((y-factor),0,image.shape[0])
        ymax = np.clip((y+factor),0,image.shape[0])
        xmin = np.clip(x-factor,0,image.shape[1])
        xmax = np.clip(x+factor,0,image.shape[1])
        return self.image[ymin:ymax,xmin:xmax]
    
    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs,image):
        outputs = outputs[self.output_name].squeeze()
        left_x = int(outputs[0]*image.shape[1])
        left_y = int(outputs[1]*image.shape[0])
        right_x = int(outputs[2]*image.shape[1])
        right_y = int(outputs[3]*image.shape[0])
        left_eye = self.clip(left_x,left_y,image)
        right_eye = self.clip(right_x,right_y,image)
        return left_eye,right_eye