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
from cv_bridge import CvBridge

#from keras.models import model_from_json

pygame.init()
size = None
width = None
height = None
screen = None

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
              sim_img = pygame.image.fromstring(img.tobytes(), size, 'RGB')

              screen.blit(sim_img, (0,0))
              pygame.display.flip()

        else:
          print "skipping ", skip, " seconds from ", t.to_sec(), " to ", skipping, " ..."
    except:
      pass
