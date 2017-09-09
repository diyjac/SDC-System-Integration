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
import sys
import numpy as np
import pygame
import rosbag
import datetime
import cStringIO
import serialrosbags
import cv2
from cv_bridge import CvBridge
import tensorflow as tf

#from keras.models import model_from_json

pygame.init()
size = None
width = None
height = None
screen = None

label = ['RED', 'YELLOW', 'GREEN', '', 'UNKNOWN']

def scale(x, feature_range=(-1, 1)):
    """Rescale the image pixel values from -1 to 1

    Args:
        image (cv::Mat): image containing the traffic light

    Returns:
        image (cv::Mat): image rescaled from -1 to 1 pixel values

    """
    # scale to (-1, 1)
    x = ((x - x.min())/(255 - x.min()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


# ***** main loop *****
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Udacity SDC Challenge-2 Video viewer 2')
  parser.add_argument('--datasets', type=str, default="dataset1.bag:dataset2.bag:dataset3.bag:dataset4.bag", help='Dataset/ROS Bag name')
  parser.add_argument('--skip', type=int, default="0", help='skip seconds')
  args = parser.parse_args()

  datasets = args.datasets
  skip = args.skip
  startsec = 0
  angle_steers = 0.0
  blue = (0, 0, 255)
  green = (0, 255, 0)

  bridge = CvBridge()
  bags = serialrosbags.Bags(datasets, ['/current_pose','/image_raw'])

  # get the traffic light classifier
  model_path = '../classifier/GAN-Semi-Supervised-site'
  config = tf.ConfigProto(log_device_placement=True)
  config.gpu_options.per_process_gpu_memory_fraction = 0.2  # don't hog all the VRAM!
  config.operation_timeout_in_ms = 50000 # terminate anything that don't return in 50 seconds
  sess = tf.Session(config=config)
  saver = tf.train.import_meta_graph(model_path + '/checkpoints/generator.ckpt.meta')
  saver.restore(sess,tf.train.latest_checkpoint(model_path + '/checkpoints/'))

  # get the tensors we need for doing the predictions by name
  tf_graph = tf.get_default_graph()
  input_real = tf_graph.get_tensor_by_name("input_real:0")
  drop_rate = tf_graph.get_tensor_by_name("drop_rate:0")
  logits = tf_graph.get_tensor_by_name("confidence:0")
  predict = tf_graph.get_tensor_by_name("predict:0")

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
              if size is None:
                size = (width, height)
                pygame.display.set_caption("Udacity SDC ROSBAG: camera video viewer")
                screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
                print "size: ", size

              img = bridge.imgmsg_to_cv2(msg, "rgb8")
              if height != 600 or width != 800:
                img2 = cv2.resize(bridge.imgmsg_to_cv2(msg, "rgb8"), (800, 600), interpolation=cv2.INTER_AREA)
              tllogits, pred = sess.run([logits, predict], feed_dict = {
                input_real: scale(img2.reshape(-1, 600, 800, 3)),
                drop_rate:0.})
              tl_logits = np.delete(tllogits[:3], np.argmin(tllogits[:3]))
              confidence = np.max(tl_logits) - np.min(tl_logits)
              if confidence < 2.:
                  pred[0] = 4
              color = (192, 192, 0)
              text = 'Most Likely: %s, confidence: %f'
              font = cv2.FONT_HERSHEY_COMPLEX
              cv2.putText(img, text%(label[pred[0]], confidence), (100, height-60), font, 1, color, 2)
              sim_img = pygame.image.fromstring(img.tobytes(), size, 'RGB')

              screen.blit(sim_img, (0,0))
              pygame.display.flip()

        else:
          print "skipping ", skip, " seconds from ", t.to_sec(), " to ", skipping, " ..."
    except:
      pass
