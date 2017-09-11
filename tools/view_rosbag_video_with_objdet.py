#!/usr/bin/python
"""
view_rosbag_video.py: version 0.1.0
Note:
Part of this code was copied and modified from github.com/comma.ai/research (code: BSD License)

Todo:
Update steering angle projection.  Current version is a hack from comma.ai's version
Update enable left, center and right camera selection.  Currently all three cameras are displayed.
Update to enable display of trained steering data (green) as compared to actual (blue projection).

History:
2016/10/06: Update to add --skip option to skip the first X seconds of data from rosbag.
2016/10/02: Initial version to display left, center, right cameras and steering angle.
"""

import argparse
import os
import sys
import numpy as np
import pygame
import rosbag
import datetime
import cStringIO
import serialrosbags
import cv2
import tensorflow as tf
from cv_bridge import CvBridge
from functools import partial

THRESHOLD = 0.50

pygame.init()
size = None
width = None
height = None
screen = None

detect_height = 400
detect_width = 600
chunksize = 1024

yellow = (255, 255, 0)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

labels = ['', 'RED', 'YELLOW', 'GREEN', '', 'UNKNOWN']
clabels = [4, 0, 1, 2, 4, 4]
colors = [blue, red, yellow, green, blue, blue]

def draw_bounding_box(img, boundingbox, color=[0,255,0], thickness=6):
  x1, y1, x2, y2 = boundingbox
  cv2.line(img, (x1, y1), (x2, y1), color, thickness)
  cv2.line(img, (x2, y1), (x2, y2), color, thickness)
  cv2.line(img, (x2, y2), (x1, y2), color, thickness)
  cv2.line(img, (x1, y2), (x1, y1), color, thickness)

def joinfiles(directory, filename, chunksize=chunksize):
    print "restoring:", filename, "from directory:", directory
    if os.path.exists(directory):
        if os.path.exists(filename):
            os.remove(filename)
        output = open(filename, 'wb')
        chunks = os.listdir(directory)
        chunks.sort()
        for fname in chunks:
            fpath = os.path.join(directory, fname)
            with open(fpath, 'rb') as fileobj:
                for chunk in iter(partial(fileobj.read, chunksize), ''):
                    output.write(chunk)
        output.close()


# ***** main loop *****
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Udacity SDC ROSBAG: camera video viewer")
  parser.add_argument('--datasets', type=str, default="dataset1.bag:dataset2.bag:dataset3.bag:dataset4.bag", help='Dataset/ROS Bag name')
  parser.add_argument('--skip', type=int, default="0", help='skip seconds')
  args = parser.parse_args()
  model_path = '../classifier/faster-R-CNN/'

  # if frozen_inference_graph.pb is not reconstituded yet, join it back together
  if not os.path.exists(model_path+'checkpoints/frozen_inference_graph.pb'):
    joinfiles(model_path+'frozen_model_chunks', model_path+'checkpoints/frozen_inference_graph.pb')

  datasets = args.datasets
  skip = args.skip
  startsec = 0
  angle_steers = 0.0

  bridge = CvBridge()
  bags = serialrosbags.Bags(datasets, ['/current_pose','/image_raw'])

  # get the traffic light classifier
  ii = 0
  config = tf.ConfigProto(log_device_placement=True)
  config.gpu_options.per_process_gpu_memory_fraction = 0.5  # don't hog all the VRAM!
  config.operation_timeout_in_ms = 50000 # terminate anything that don't return in 50 seconds
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path+'checkpoints/frozen_inference_graph.pb', 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
      with tf.Session(graph=detection_graph, config=config) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # loop through and make our predictions
        while bags.has_data():
          try:
            topic, msg, t = bags.read_messages()
            if startsec == 0:
              startsec = t.to_sec()
              if skip < 24*60*60:
                skipping = t.to_sec() + skip
                print "skipping ", skip, " seconds from ", startsec, " to ", skipping, " ..."
              else:
                skipping = skip
                print "skipping to ", skip, " from ", startsec, " ..."
            else:
              if t.to_sec() > skipping:
                if topic in ['/current_pose', '/image_raw']:
                  blue = (255, 0, 0)
                  print(topic, msg.header.seq, t-msg.header.stamp, t)
      
                  if topic == '/image_raw':

                    height = int(msg.height)
                    width = int(msg.width)
                    detect_height = height
                    detect_width = width
                    image = bridge.imgmsg_to_cv2(msg, "rgb8")
                    if height != detect_height or width != detect_width:
                      image_np = cv2.resize(image, (detect_width, detect_height), interpolation=cv2.INTER_AREA)
                    else:
                      image_np = np.copy(image)

                    if size is None:
                      size = (width, height)
                      pygame.display.set_caption("Udacity SDC ROSBAG: camera video viewer")
                      screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
                      print "size: ", size

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    # Actual detection
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # Visualization of the results of a detection.
                    classes = np.squeeze(classes).astype(np.int32)
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)

                    c = 5
                    textlabel = labels[c]
                    label = clabels[c]
                    color = colors[c]
                    cc = classes[0]
                    confidence = scores[0]
                    if cc > 0 and cc < 4 and confidence is not None and confidence > THRESHOLD:
                      c = cc
                      textlabel = labels[c]
                      label = clabels[c]
                      color = colors[c]
                      x1 = int(width * boxes[0][1])
                      y1 = int(height * boxes[0][0])
                      x2 = int(width * boxes[0][3])
                      y2 = int(height * boxes[0][2])

                    text0 = 'Frame: %d'
                    text1 = 'Most Likely: %s'
                    text2 = 'Confidence: %f'
                    font = cv2.FONT_HERSHEY_COMPLEX
                    cv2.putText(image, text0%(ii), (30, height-90), font, 1, color, 1)
                    cv2.putText(image, text1%(textlabel), (30, height-60), font, 1, color, 1)
                    if c > 0 and c < 4:
                      cv2.putText(image, text2%(confidence), (30, height-30), font, 1, color, 1)
                      draw_bounding_box(image, (x1, y1, x2, y2), color=color, thickness=1)
                    sim_img = pygame.image.fromstring(image.tobytes(), size, 'RGB')

                    screen.blit(sim_img, (0,0))
                    pygame.display.flip()
                    ii += 1

              else:
                print "skipping ", skip, " seconds from ", t.to_sec(), " to ", skipping, " ..."
          except:
            pass
