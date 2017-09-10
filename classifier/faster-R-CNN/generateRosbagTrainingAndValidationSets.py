#!/usr/bin/env python

import argparse
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import six

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import dataset_util

textlabel = [ 'RED', 'YELLOW', 'GREEN' ]
label = [ 1, 2, 3 ]

def encode_image_array_as_jpg_str(image):
  """Encodes a numpy array into a JPEG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    JPEG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='JPEG')
  jpg_string = output.getvalue()
  output.close()
  return jpg_string

def create_record(box, newclass, image, filepath):
    height = image.shape[0]
    width = image.shape[1]
    filename = filepath
    encoded_image_data = encode_image_array_as_jpg_str(image)
    image_format = b'jpeg'

    xmins = [ box[1] ] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [ box[3] ] # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [ box[0] ] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [ box[2] ] # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [ textlabel[newclass] ] # List of string class name of bounding box (1 per box)
    classes = [ label[newclass] ] # List of integer class id of bounding box (1 per box)

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_record

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the model that is used for finding the bounding box
# of the traffic light in the image
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_PRETRAIN_LABELS = './models/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90
TRAFFIC_LIGHT_CLASS = 10
THRESHOLD = 0.89
PATH_TO_NEW_LABLES = './tl_label_map.pbtxt'

# already load pre-train model and weights?
if not os.path.exists('faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.index'):
    if not os.path.exists('faster_rcnn_resnet101_coco_11_06_2017.tar.gz'):
        print "Downloading Pre-trained model and weights..."
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    print "Extracting Pre-trained model and weights..."
    for file in tar_file.getmembers():
        tar_file.extract(file, os.getcwd())
    print "Done!"

# load the model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_PRETRAIN_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


if __name__ == '__main__':
    defaultOutput = 'tl_train.record'
    parser = argparse.ArgumentParser(description='Udacity SDC: System Integration - Traffic Light CSV to TFRecord file format converter')
    parser.add_argument('--infilename', type=str, default='loop_with_traffic_light.csv', help='traffic light csv input file')
    parser.add_argument('--outfilename', type=str, default=defaultOutput, help='TFRecord file')
    args = parser.parse_args()
    incsv = os.getcwd()+'/../../test_images/'+args.infilename

    # read in incsv file
    imagefiles = []
    newlabels = []
    with open(incsv) as f:
        count = 0
        for line in f:
            if count > 0:
                data = line.split(',')
                imagefiles.append(os.getcwd()+'/../../'+data[7].replace("'", ""))
                newlabels.append(int(data[8]))
            count += 1

    # start writing to the record file
    writer = tf.python_io.TFRecordWriter(args.outfilename)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for i in range(len(imagefiles)):
                if newlabels[i] < 3:     # don't try to classify 'unknown'
                    image = Image.open(imagefiles[i])
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    classes = np.squeeze(classes).astype(np.int32)
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)
                    for j in range(len(classes)):
                        if classes[j] == TRAFFIC_LIGHT_CLASS:
                            if scores[j] is not None and scores[j] > THRESHOLD:
                                print "writing record:", i, imagefiles[i], boxes[j], "score:", scores[j], "new label:", newlabels[i]
                                record = create_record(boxes[j], newlabels[i], image_np, imagefiles[i])
                                writer.write(record.SerializeToString())
    writer.close()

