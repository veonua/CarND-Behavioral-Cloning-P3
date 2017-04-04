# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_writeup.md_summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I've used nvidia model as relatively simple an efficient for this task

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 90, 320, 3)        12        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              9834636   
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================


#### 2. Attempts to reduce overfitting in the model

For dataset I tried to record "important" parts of the track, like turns, hills. As dataset was shuffled and normilized there is no-or-little overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 47). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Also I tried to smooth angle values using running average function. But it caused troubles on second track.

#### 3. Model parameter tuning

The model used an adam optimizer, but for faster convergence initial learning rate was doubled (line 116).
Loss function was changed to "mean absolute error" because steering values are less than 1.0. So mean squared error caused slower covergence.

Also I've tried different color schemas. The version with only gray channel made it pretty good, it was able to make 80% of track, but on second track (the one that was not in training set) it was not so successful. 

So I've ended up with YUV channels. I believe model has learned that car must stay on area with low saturation (gray road) and near bright lines

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road, and to have more interesting data in the set (curves, hills, ets). 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First of all I studied available architectures for this task.

For the beginning I've used basic architecture, just to verify that all systems work properly. So I started with couple of convolution layers followed by coulple full-connected layers. And suprisingly it was a pretty good start, car managed to make through several turns and even get to the bridge.

Then I've played with VGG but it was pretty slow. And I made a *HUGE* mistake I put image processing only to model.py and did not applied processing to drive.py file. So next several hours were wasted trying to understand why better loss function gives worse results on the track. 

So I swithed to Nvidia model and started over. Nvidia model on RGB data able to make through Track1 but was pretty useless on second track, so I decided to play with image processing again in order to achive more general model. I've tried histogram normalization and augmentation but it gave almost no gain in quality of driving probably because Nvidia model has batch normalization as first layer. So I switched to YUV and increased number of epochs

At the end of the process, the vehicle is able to drive autonomously around on both tracks without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 87-108) 

Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 90, 320, 3)        12        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              9834636   
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================

see https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track using center lane driving. I used left and right camera images as "recovery driving" dataset, also flipped images all these and angles thinking that this would give more data points. 

After the collection process, I had ~47292 number of data points. I then preprocessed this data by crop and color-space conversion.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 23, I've used early stopping with {monitor='val-loss', min_delta=0.005, patience=5} parameters. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
