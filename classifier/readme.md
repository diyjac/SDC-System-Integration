# Traffic Classifier README.md
---
This document describes the current status of traffic lights classification uses for the capstone project Udacity SDC. 
This is the status quo as of August 24th, 2017 with almost no stable internet connection, so a number of operations, like getting new data are not feasible yet. 

## Working Hack 

As time is short and we want to progress fast I compiled this section more or less bluntly copying this [code](https://github.com/mynameisguy/TrafficLightChallenge-DeepLearning-Nexar) to enable implementation of a working traffic lights classifier using the following steps:

1. Download [weights & code](https://github.com/mynameisguy/TrafficLightChallenge-DeepLearning-Nexar/tree/master) of this [pretrained squezzenet model](https://github.com/davidbrai/deep-learning-traffic-lights)

2. Use the following code to predict an image

It follows "pred.py": 
 
    # imports
    import os
    from keras.preprocessing.image import load_img,img_to_array
    import numpy as np
    from train import SqueezeNet
    from consts import IMAGE_WIDTH, IMAGE_HEIGHT
    
    # load pretrained model 
    model = SqueezeNet(3, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model.load_weights("challenge1.weights")
    
    # predict image class
    filepath = "./02-image_from_simulator/sampleout1.jpg"
    image = load_img(filepath, target_size=(224,224))
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
    
I guess we have to train our own model which can be done in the meantime together tweaking the hyper parameters. 

###### Or are we allowed to use other models if we cite them properly?

## 1. Datasets

### 1.1 Bosch Small Traffic Lights Dataset (Sebastian)

https://hci.iwr.uni-heidelberg.de/node/6132
https://hci.iwr.uni-heidelberg.de/node/6132/download/a96be3a973d1ba808830d8c445c62efd (download)


### 1.2 LISA Traffic Light Dataset (Sebastian)

http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/
http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/traffic-light-detection/ (download)
http://cvrr.ucsd.edu/vivachallenge/index.php/signs/sign-detection/ (workflow)


### 1.3 Images from a [ROS Traffic Light Classifier](https://github.com/cena0805/ros-traffic-light-classifier)

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
S
### 1.4 [Udacity Camera Feed](https://carnd.slack.com/archives/C6NVDVAQ3/p1503614726000196)

https://drive.google.com/open?id=0B2_h37bMVw3iYkdJTlRSUlJIamM

Caleb Kirksey provided us with two bag files with an image feed from the Udacity self-driving car's camera in the test lot and a topic containing the car's position at [google drive](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing).

Caleb Kirksey: "The video has some shots of the traffic light that we'll be using in testing. I'll follow up with the relative location of the traffic light as well as more images of the light for training a neural net."

## 2. Neural Net Candidates

## 2.1 Traffic Light Color Classifier (Rainer)

This is a small Jupiter notebook for testing the approach of [this ROS node] (https://github.com/cena0805/ros-traffic-light-classifier).

Description see 4.1 deadends.

It contains all data preparation and a NOT working version of squeeze net. NEEDS ATTENTION!!!

|no. |lr    | batch | images | model     | Opt.    | epochs | s/epoch | loss           acc        | test acc. |   
|:--:|:----:|:-----:|:------:|:---------:|:-------:|:------:|:-------:|:-----------------:|:--------:| 
| 001 |--- |   64  |  all    | get_model | RMSPROP | 25      |600s    | loss: 0.25   - acc: 0.85,  |>>> 44%|
| 002 |0.0001| 64  |  2000   | get_model | SGD     | 25      | 50s    | loss: 0.3328 - acc: 0.8896 |>>> 42.4%|
|003 |0.001 | 64   | 2000   | get_model | SGD     | 25     |  50s    | loss: 0.1373 - acc: 0.9585 | >>> 48.4% / 49.6%|
|004 |0.001 | 16   | 2000   | get_model | SGD     | 25     |  80s    | loss: 0.1613 - acc: 0.9545 |>>> 47.5% / 50.1%|
|005| 0.001 | 128  | 2000   | get_model | SGD     | 25     |  50s    | loss: 0.1627 - acc: 0.9521 |>>> 42.9%|
|006 |0.005 | 64   | 2000   | get_model | SGD     | 25      | 50s    | loss: 0.1304 - acc: 0.9673 |>>> 44.3%|
|007 |0.001 | 64    |all    | get_model | SGD     | 50     |  850s   | loss: 0.0919 - acc: 0.9737 |>>> 50.0% <<<|
test accuracy is determined for predicting a completly unseen test dataset splitted from the original dataset


## 2.2 Using [Yolo v2](https://github.com/chrisgundling/yolo_light)

##### Testing on test traffic lights
    1. install yolo according to link above
    2. Copy files into folder ./test
    3. ./flow --test /Users/rainerbareiss/Downloads/traffic_light_images/PATH --model cfg/tiny-yolo-udacity.cfg --load 8987 --json
    4. ./flow --test test/ --model cfg/tiny-yolo-udacity.cfg --load 8987
    

Other link to yolo is https://github.com/udacity/self-driving-car/tree/master/vehicle-detection/darkflow/net. This is based on pytho3 and tensor flow 1.2.


## 2.3 Pretrained [Squeezenet](https://github.com/davidbrai/deep-learning-traffic-lights)

The approach of the Winner of the Nexar Competition is described very extensively together with alternative methods that worked out and did not work out in [this article](
https://medium.freecodecamp.org/recognizing-traffic-lights-with-deep-learning-23dae23287cc). 

The complete code - without the data! - is on [github](https://github.com/davidbrai/deep-learning-traffic-lights).

##### other links
1. Separate pretained https://github.com/rcmalli/keras-squeezenet ok!

2. [Other Nexar Code](https://github.com/mynameisguy/TrafficLightChallenge-DeepLearning-Nexar). This model was created for the nexar challenge 1 and reached 93.6% accuracy on the challenge test data. This work is based on the squeezeNet model and it's keras implementation. 
######TODO: check this!!!!!

## 2.4 GAN

Needs Attention

## 2.5 Training a [Haar Classifier](http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html)

http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html

## 3 Finetuning Neural Nets

### 3.1 Keras: Guide from medium 

[A Comprehensive guide to Fine-tuning Deep Learning Models in Keras (Part I)](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html)

[A Comprehensive guide to Fine-tuning Deep Learning Models in Keras (Part II)](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)


### 3.2 Caffe according to google group [Caffe-users](https://groups.google.com/forum/#!topic/caffe-users/wWSGX4vmAh4)

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


## 4 Deadends

### 4.1 Color Classifier

Applying a [color classifier](https://github.com/algolia/color-extractor) from github just detected a lot of black and grey and no prominent green or red in the images of the dataset nr. 1.3.

### 4.2 Neural Net Nr. 2.1

Downloading the original trained network and applying it on the test data set show almost the same accuracy of 50% over all classes. Thats exact the same accuracy I was ending. The strange thin is that the training accuracy was 97%. I must be overlooking a very obvious fact.

For completeness here the results:

|Accuracy |all      | red      | green   | unknown  |
|:-------:|:-------:|:--------:|:-------:|:--------:|
|for predictinon on test data| 50.83%| 50.17%  | 50.82%  | 51.15%   |

The structure of the net is

    ____________________________________________________________________________________________________
    
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    convolution2d_5 (Convolution2D)  (None, 64, 64, 32)    896         convolution2d_input_6[0][0]      
    ____________________________________________________________________________________________________
    activation_7 (Activation)        (None, 64, 64, 32)    0           convolution2d_5[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_6 (Convolution2D)  (None, 62, 62, 32)    9248        activation_7[0][0]               
    ____________________________________________________________________________________________________
    activation_8 (Activation)        (None, 62, 62, 32)    0           convolution2d_6[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_3 (MaxPooling2D)    (None, 31, 31, 32)    0           activation_8[0][0]               
    ____________________________________________________________________________________________________
    dropout_4 (Dropout)              (None, 31, 31, 32)    0           maxpooling2d_3[0][0]             
    ____________________________________________________________________________________________________
    convolution2d_7 (Convolution2D)  (None, 31, 31, 64)    18496       dropout_4[0][0]                  
    ____________________________________________________________________________________________________
    activation_9 (Activation)        (None, 31, 31, 64)    0           convolution2d_7[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_8 (Convolution2D)  (None, 29, 29, 64)    36928       activation_9[0][0]               
    ____________________________________________________________________________________________________
    activation_10 (Activation)       (None, 29, 29, 64)    0           convolution2d_8[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_4 (MaxPooling2D)    (None, 14, 14, 64)    0           activation_10[0][0]              
    ____________________________________________________________________________________________________
    dropout_5 (Dropout)              (None, 14, 14, 64)    0           maxpooling2d_4[0][0]             
    ____________________________________________________________________________________________________
    flatten_2 (Flatten)              (None, 12544)         0           dropout_5[0][0]                  
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 512)           6423040     flatten_2[0][0]                  
    ____________________________________________________________________________________________________
    activation_11 (Activation)       (None, 512)           0           dense_3[0][0]                    
    ____________________________________________________________________________________________________
    dropout_6 (Dropout)              (None, 512)           0           activation_11[0][0]              
    ____________________________________________________________________________________________________
    dense_4 (Dense)                  (None, 3)             1539        dropout_6[0][0]                  
    ____________________________________________________________________________________________________
    activation_12 (Activation)       (None, 3)             0           dense_4[0][0]                    
    ====================================================================================================
    Total params: 6,490,147
    Trainable params: 6,490,147
    Non-trainable params: 0
    ____________________________________________________________________________________________________

## 5 Interesting Links
[Rescale](http://www.rescale.com/?_ga=2.243752647.319079710.1503424790-1699517006.1503424790) a powerful cloud infrastructure e.g. for machine learning

[DenseNet on medium](https://medium.com/@ManishChablani/densenet-2810936aeebb?source=userActivityShare-5cf6967007c8-1503652277) and on [github](https://github.com/flyyufelix/DenseNet-Keras/blob/master/README.md)