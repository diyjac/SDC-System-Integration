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

bridge = CvBridge()
bags = None

def make_frame(t):
  global bridge
  global bags
  while bags.has_data():
    try:
      topic, msg, t = bags.read_messages()
      if topic == '/image_raw':
        return cv2.resize(bridge.imgmsg_to_cv2(msg, "rgb8"), (800, 600), interpolation=cv2.INTER_AREA)
    except:
      pass
  return None

# ***** main loop *****
if __name__ == "__main__":
  defaultOutput = 'out.mp4'
  parser = argparse.ArgumentParser(description='Udacity SDC Challenge-2 Video viewer 2')
  parser.add_argument('--datasets', type=str, default="dataset1.bag:dataset2.bag:dataset3.bag:dataset4.bag", help='Dataset/ROS Bag name')
  parser.add_argument('outfilename', type=str, default=defaultOutput, help='video output file')
  args = parser.parse_args()
  videopattern = re.compile("^.+\.mp4$")

  if videopattern.match(args.outfilename):
    if os.path.exists(args.outfilename):
      print("Video output file: %s exists.  %s" % (
        args.outfilename, pleaseRemove))
      sys.exit(2)
    else:
      videoout = args.outfilename
      valid = True
  else:
    print(invalidExt % ("video", validVideoExt))
    sys.exit(3)

  datasets = args.datasets
  bags = serialrosbags.Bags(datasets, ['/current_pose','/image_raw'])

  clip = mpy.VideoClip(make_frame, duration=14)
  clip.write_videofile(videoout, fps=24, audio=False)  

