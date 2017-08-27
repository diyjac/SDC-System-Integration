from styx_msgs.msg import TrafficLight
# imports
import os
from keras.preprocessing.image import load_img,img_to_array
from keras import backend as K
import tensorflow as tf
import numpy as np
import cv2
from train import SqueezeNet
from consts import IMAGE_WIDTH, IMAGE_HEIGHT

class TLClassifier(object):
    def __init__(self):
        # parameters
        self.tf_session = K.get_session()
        self.tf_graph = self.tf_session.graph

        # load pretrained model 
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                self.model = SqueezeNet(3, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
                self.model.load_weights("challenge1.weights")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # predict image class
        #filepath = "./02-image_from_simulator/sampleout1.jpg"
        #image = load_img(filepath, target_size=(224,224))
        image = img_to_array(cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA))
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                preds = self.model.predict(image)[0]
        predclass = str(np.argmax(preds))
        print(predclass)
        ##predclass = 2
        print("hello tl_classifier")
        # 
        # predclass
        # 0: no traffic light
        # 1: red light
        # 2: green light
        if (predclass == 1):
          return TrafficLight.RED
        elif (predclass == 2):
          return TrafficLight.GREEN
        elif (predclass == 3):
          return TrafficLight.YELLOW
        else:    
          return TrafficLight.UNKNOWN

#uint8 UNKNOWN=4
#uint8 GREEN=2
#uint8 YELLOW=1
#uint8 RED=0
