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
import re
import os
import cv2
import numpy as np
import rosbag
import datetime
import serialrosbags
from cv_bridge import CvBridge
import moviepy.editor as mpy
import csv

bridge = CvBridge()
bags = None
i = 0
pose = None

# ***** main loop *****
if __name__ == "__main__":
  defaultOutput = 'out%04d.jpg'
  parser = argparse.ArgumentParser(description='Udacity SDC Challenge-2 Video viewer 2')
  parser.add_argument('--datasets', type=str, default="dataset1.bag:dataset2.bag:dataset3.bag:dataset4.bag", help='Dataset/ROS Bag name')
  parser.add_argument('outfilename', type=str, default=defaultOutput, help='jpeg output file pattern')
  args = parser.parse_args()
  picturepattern = re.compile("^.+\.jpg$")

  fieldname = ['x', 'y', 'z', 'ax', 'ay', 'az', 'aw', 'image', 'label']
  log_file = open('out.csv', 'w')
  log_writer = csv.DictWriter(log_file, fieldnames=fieldname)
  log_writer.writeheader()

  jpgout = args.outfilename
  datasets = args.datasets
  bags = serialrosbags.Bags(datasets, ['/current_pose','/image_raw'])
  while bags.has_data():
    try:
      topic, msg, t = bags.read_messages()
      if topic == '/current_pose':
        pose = msg
      if topic == '/image_raw':
        image = cv2.resize(bridge.imgmsg_to_cv2(msg, "rgb8"), (800, 600), interpolation=cv2.INTER_AREA)
        print "writing file", jpgout%(i)
        if pose is not None:
          log_writer.writerow({
            'x': pose.pose.position.x,
            'y': pose.pose.position.y,
            'z': pose.pose.position.z,
            'ax': pose.pose.orientation.x,
            'ay': pose.pose.orientation.y,
            'az': pose.pose.orientation.z,
            'aw': pose.pose.orientation.w,
            'image': "'"+jpgout%(i)+"'",
            'label': 0})
        cv2.imwrite(jpgout%(i), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        i += 1
    except:
      pass

  log_file.close()

