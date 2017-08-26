#!/usr/bin/python
"""
serialrosbags.py: version 0.2.0

History:
2016/10/13: Initial version stream multiple rosbags in sync.
"""

import sys
import rosbag

class Bags():
  bags = []
  nextmsgIdx = []
  nextmsg = []
  currentidx = -1
  currentmsg = {}
  currentbag = ""
  currenttime = -1
  mytopics = []

  def __init__(self, givendatasets, giventopics=None):
    Bags.datasets = givendatasets.split(':')
    Bags.mytopics = giventopics

    ## Initialize bags
    i = 0
    for dataset in Bags.datasets:

      #initialize item bag
      Bags.nextmsgIdx[i:] = [ 0 ]
      Bags.nextmsg[i:] = [ 0 ]

      print "opening ", dataset
      bag = rosbag.Bag(dataset, 'r')

      newbag = []
      # load the dataset
      print "reading ", dataset
      for topic, msg, t in bag.read_messages(Bags.mytopics):
        newbag[len(newbag):] = [ {'topic':topic, 'msg':msg, 't':t} ]
      bag.close()

      # process initial
      Bags.bags[i:] = [ newbag ]
      Bags.nextmsgIdx[i] = 0
      Bags.nextmsg[i] = ( Bags.bags[i][Bags.nextmsgIdx[i]]['topic'], Bags.bags[i][Bags.nextmsgIdx[i]]['msg'], Bags.bags[i][Bags.nextmsgIdx[i]]['t'] )
      if Bags.currenttime > 0:
        if Bags.currenttime > t.to_sec():
          Bags.currenttime = t.to_sec()
          Bags.currentmsg = ( Bags.bags[i][Bags.nextmsgIdx[i]]['topic'], Bags.bags[i][Bags.nextmsgIdx[i]]['msg'], Bags.bags[i][Bags.nextmsgIdx[i]]['t'] )
          Bags.currentidx = i
      else:
        Bags.currenttime = Bags.bags[i][Bags.nextmsgIdx[i]]['t'].to_sec()
        Bags.currentmsg = ( Bags.bags[i][Bags.nextmsgIdx[i]]['topic'], Bags.bags[i][Bags.nextmsgIdx[i]]['msg'], Bags.bags[i][Bags.nextmsgIdx[i]]['t'] )
        Bags.currentidx = i
      i += 1

    # grab a new bag element for the selected bag that is the very first
    Bags.nextmsgIdx[Bags.currentidx] += 1
    Bags.nextmsg[Bags.currentidx] = ( Bags.bags[Bags.currentidx][Bags.nextmsgIdx[Bags.currentidx]]['topic'], \
                                      Bags.bags[Bags.currentidx][Bags.nextmsgIdx[Bags.currentidx]]['msg'], \
                                      Bags.bags[Bags.currentidx][Bags.nextmsgIdx[Bags.currentidx]]['t'] )

  def has_data(self):
    for i in range(len(Bags.bags)):
      if Bags.nextmsgIdx[i] >= len(Bags.bags[i]):
        return False
    return True

  def read_messages(self, topics=None):
    ret = Bags.currentmsg
    if topics is None:
       topics = Bags.mytopics

    # find the next topic in time
    nexttime = -1
    for i in range(len(Bags.bags)):
      if nexttime > 0:
         if nexttime > Bags.nextmsg[i][2].to_sec():
           nexttime = Bags.nextmsg[i][2].to_sec()
           Bags.currentbag = Bags.bags[i]
           Bags.currentmsg = Bags.nextmsg[i]
           Bags.currentidx = i
      else:
         nexttime = Bags.nextmsg[i][2].to_sec()
         Bags.currenttime = nexttime
         Bags.currentbag = Bags.bags[i]
         Bags.currentmsg = Bags.nextmsg[i]
         Bags.currentidx = i

    # grab a new bag element for the selected bag
    Bags.nextmsgIdx[Bags.currentidx] += 1
    Bags.nextmsg[Bags.currentidx] = ( Bags.bags[Bags.currentidx][Bags.nextmsgIdx[Bags.currentidx]]['topic'], \
                                      Bags.bags[Bags.currentidx][Bags.nextmsgIdx[Bags.currentidx]]['msg'], \
                                      Bags.bags[Bags.currentidx][Bags.nextmsgIdx[Bags.currentidx]]['t'] )

    # return the current message from the beginning
    return ret

