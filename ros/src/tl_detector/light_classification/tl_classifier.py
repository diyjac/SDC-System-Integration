from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        # imports
        import os
        from keras.preprocessing.image import load_img,img_to_array
        import numpy as np
        from train import SqueezeNet
        from consts import IMAGE_WIDTH, IMAGE_HEIGHT
        
        # load pretrained model 
        model = SqueezeNet(3, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        model.load_weights("challenge1.weights")

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
        image = img_to_array(image)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        preds = model.predict(image)[0]
        predclass = str(np.argmax(preds))
        print(predclass)
        # 
        # predclass
        # 0: no traffic light
        # 1: red light
        # 2: green light
        if (predclass == 1):
          return TrafficLight.RED
        elif (predclass == 2):
          return TrafficLight.GREEN
        elif (predclass == 3)
          return TrafficLight.YELLOW
        else:    
          return TrafficLight.UNKNOWN

#uint8 UNKNOWN=4
#uint8 GREEN=2
#uint8 YELLOW=1
#uint8 RED=0