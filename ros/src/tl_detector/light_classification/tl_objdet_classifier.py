from styx_msgs.msg import TrafficLight

import os
import sys
import tensorflow as tf
import numpy as np
from functools import partial

THRESHOLD = 0.50

class TLODClassifier(object):
    def __init__(self, model_path):
        #DONE load classifier model and restore weights
        self.tf_session = None
        self.predict = None
        self.model_path = model_path
        self.clabels = [4, 0, 1, 2, 4, 4]
        self.readsize = 1024
        # was model and weights made whole?
        if not os.path.exists(self.model_path+'/checkpoints/frozen_inference_graph.pb'):
            # if not - build it back up
            if not os.path.exists(self.model_path+'/chunks'):
                output = open(self.model_path+'/checkpoints/frozen_inference_graph.pb', 'wb')
                chunks = os.listdir(self.model_path+'/frozen_model_chunks')
                chunks.sort()
                for filename in chunks:
                    filepath = os.path.join(self.model_path+'/checkpoints', filename)
                    with open(filepath, 'rb') as fileobj:
                        for chunk in iter(partial(fileobj.read, self.readsize), ''):
                            output.write(chunk)
                output.close()

    def get_classification(self, image_np):
        """Determines the color of the traffic light in the image

        Args:
            image_np (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #DONE implement light color prediction
        # Our final model is based on Tensorflow Model Object Detection API classifier.
        # We do not have to tranform the image to the traffic light, because this
        # Object Detection classifier will classify the entire image.

        # set up tensorflow and traffic light classifier
        if self.tf_session is None:
            # get the traffic light classifier
            self.config = tf.ConfigProto(log_device_placement=True)
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.5  # don't hog all the VRAM!
            self.config.operation_timeout_in_ms = 50000 # terminate anything that don't return in 50 seconds
            self.tf_graph = tf.Graph()
            with self.tf_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path+'/checkpoints/frozen_inference_graph.pb', 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                    self.tf_session = tf.Session(graph=self.tf_graph, config=self.config)
                    # Definite input and output Tensors for self.tf_graph
                    self.image_tensor = self.tf_graph.get_tensor_by_name('image_tensor:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    self.detection_scores = self.tf_graph.get_tensor_by_name('detection_scores:0')
                    self.detection_classes = self.tf_graph.get_tensor_by_name('detection_classes:0')
                    self.num_detections = self.tf_graph.get_tensor_by_name('num_detections:0')
                    self.predict = True

        predict = TrafficLight.UNKNOWN
        if self.predict is not None:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Actual detection
            (scores, classes, num) = self.tf_session.run(
                [self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            # calculate prediction
            c = 5
            predict = self.clabels[c]
            cc = classes[0]
            confidence = scores[0]
            if cc > 0 and cc < 4 and confidence is not None and confidence > THRESHOLD:
                c = cc
                predict = self.clabels[c]
        return predict

