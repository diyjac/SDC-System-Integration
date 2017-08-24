# Traffic Classifier README.md
---
This document describes the current status of traffic lights classification uses for the capstone project Udacity SDC. 
This is the status quo as of August 24th, 2017 with almost no stable internet connection, so a number of operations, like getting new data are not feasible yet. 

Rainer Bareiß, V0.1

##1. Datasets

###1.1 Bosch Small Traffic Lights Dataset (Sebastian)

https://hci.iwr.uni-heidelberg.de/node/6132
https://hci.iwr.uni-heidelberg.de/node/6132/download/a96be3a973d1ba808830d8c445c62efd (download)


###1.2 LISA Traffic Light Dataset (Sebastian)

http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/
http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/traffic-light-detection/ (download)
http://cvrr.ucsd.edu/vivachallenge/index.php/signs/sign-detection/ (workflow)


###1.3 Images from a [ROS Traffic Light Classifier](https://github.com/cena0805/ros-traffic-light-classifier)

These [images are cropped](https://github.com/cena0805/ros-traffic-light-classifier/tree/master/model) around several  traffic lights.

| PATH           | samples | detected by yolo |
|:--------------:|:-------:|:----------------:|
| /train/red     | 7133    | ---                 |
| /train/green   | 4638    | ---                 |
| /train/unknown | 23201   | ---                 |
| /test/red      | 1759    |                  |
| /test/green    | 1165    |                  |
| /test/unknown  | 5823    |                  |

I believe the problem with this dataset is the high amount of data in the "unknown" class which will make the neural net learn a lot of unnecessary characteristics.

##2. Neural Net Candidates

##2.1 Traffic Light Color Classifier (Rainer)

This is a small Jupiter notebook for testing the approach of [this ROS node] (https://github.com/cena0805/ros-traffic-light-classifier).

It contains all data preparation and a NOR working version of squeeze net. NEEDS ATTENTION!!!

##2.2 Using [Yolo v2](https://github.com/chrisgundling/yolo_light)

##### Testing on test traffic lights
    1. install yolo according to link above
    2. Copy files into folder ./test
    3. ./flow --test /Users/rainerbareiss/Downloads/traffic_light_images/PATH --model cfg/tiny-yolo-udacity.cfg --load 8987 --json
    4. ./flow --test test/ --model cfg/tiny-yolo-udacity.cfg --load 8987
    

Other link to yolo is https://github.com/udacity/self-driving-car/tree/master/vehicle-detection/darkflow/net. This is based on pytho3 and tensor flow 1.2.


##2.3 Pretrained [Squeezenet](https://github.com/davidbrai/deep-learning-traffic-lights)

The approach of the Winner of the Nexar Competition is described very extensively together with alternative methods that worked out and did not work out in [this article](
https://medium.freecodecamp.org/recognizing-traffic-lights-with-deep-learning-23dae23287cc). 

The complete code - without the data! - is on [github](https://github.com/davidbrai/deep-learning-traffic-lights).

#####other links
1. Separate pretained https://github.com/rcmalli/keras-squeezenet ok!

2. [Other Nexar Code](https://github.com/mynameisguy/TrafficLightChallenge-DeepLearning-Nexar). This model was created for the nexar challenge 1 and reached 93.6% accuracy on the challenge test data. This work is based on the squeezeNet model and it's keras implementation. 
######TODO: check this!!!!!

##2.4 GAN

Needs Attention

##2.5 Training a [Haar Classifier](http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html)

http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html

##3 Finetuning Neural Nets

###3.1 Keras: Guide from medium 

[A Comprehensive guide to Fine-tuning Deep Learning Models in Keras (Part I)](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html)

[A Comprehensive guide to Fine-tuning Deep Learning Models in Keras (Part II)](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)


###3.2 Caffe according to google group [Caffe-users](https://groups.google.com/forum/#!topic/caffe-users/wWSGX4vmAh4)

    ###########################
    # Predict using Caffe model
    ###########################
    Make sure that caffe is on the python path:
    caffe_root = '/home/user/CODE/caffe/'  # this file is expected to be in {caffe_root}/examples
    import sys
    sys.path.insert(0, caffe_root + 'python')
    
    import caffe
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '/home/melgor/CODE/caffe/examples/Try/train_val.prototxt'
    PRETRAINED = '/home/melgor/CODE/caffe/examples/Try/_iter_45000.caffemodel'
    
    net = caffe.Net (MODEL_FILE,PRETRAINED)
    net.set_phase_test()
    net.set_mode_cpu()
    data4D = np.zeros([100,1,1,3593]) #create 4D array, first value is batch_size, last number of input
    data4DL = np.zeros([100,1,1,1])	  # need to create 4D array as output, first value is batch_size, last number of output
    data4D[0:100,0,0,:] = xtest[0:100,:] # fill value of input
    net.set_input_arrays(data4D.astype(np.float32),data4DL.astype(np.float32))
    pred = net.forward()


##4 Deadends

###4.1 Color Classifier

Applying a [color classifier](https://github.com/algolia/color-extractor) from github just detected a lot of black and grey and no prominent green or red in the images of the dataset nr. 1.3.

###4.2 Neural Net Nr. 2.1

Downloading the original trained network and applying it on the test data set show almost the same accuracy of 50% over all classes. Thats exact the same accuracy I was ending. The strange thin is that the training accuracy was 97%. I must be overlooking a very obvious fact.

For completeness here the results:

|Accuracy |all      | red      | green   | unknown  |
|:-------:|:-------:|:--------:|:-------:|:--------:|
|for predictinon on test data| 50.83%| 50.17%  | 50.82%  | 51.15%   |



##5 Interesting Links
[Rescale](http://www.rescale.com/?_ga=2.243752647.319079710.1503424790-1699517006.1503424790) a powerful cloud infrastructure e.g. for machine learning