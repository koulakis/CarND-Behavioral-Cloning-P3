# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[model_layers]: ./model.png "Model Layers"
[track_1_fast]: ./video_first_track_fast.mp4 "Track 1 Fast"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py`: containing the script to create and train the model
* `drive.py`: for driving the car in autonomous mode
* `model.h5`: containing a trained convolution neural network 
* `writeup_report.md`: summarizing the results

It additionally contains the following files:
* `clone.ipynb`: which is an experimentation notebook used to develop and fine tune the model
* `cloning`: is a small python project which contains:
    * `image_generator an image`: a generator which loads data and applies augmentation
    * `visualizations`: visualization of convolution layers
* `*.sh`: several shell scripts which provide a quick & dirty way of setting up and controlling a remote workspace in AWS

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The `drive.py` has been edited in order to change the car speed from `9` to `30.19` which is the maximum one.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Additionally, the `clone.ipynb` has more details on the different parts of the project and contains visualization of the network layers as well as a short analysis of the predictions on the test dataset.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model architecture is exactly the one described in the [nvidia paper](https://arxiv.org/pdf/1604.07316v1.pdf). It is composed of:
- 5 convolutional layers which output 24, 36, 48, 64, 64 channels. The first three also reduce the size of the image by using a stride value of 2.
- 4 dense layers with 100, 50, 10, 1 neurons. The last one produces a linear output.

After each layer batch normalization layer is added in order to avoid over-fitting and a ReLu activation function is applied to introduce non-linearity.

#### 2. Attempts to reduce overfitting in the model

The most stable and effective method to avoid overfitting was [batch normalization](https://en.wikipedia.org/wiki/Batch_normalization). Though dropout was also tested in several experiments (also combined with batch normalization), it needed manual tuning after each architecture change and reduced the network performance a bit. 

In order to detect overfitting, two additional datasets were generated on the race track, `validation_rounds` and `test_rounds` which were used to check the performance of fine tuning and the final performance respectively.

Though the learning curves of both the testing and validation sets are really close, it is hard to say if the model is over-fitting of the specific track. The model fails on the second track which is unfortunately very different in both texture and turn sharpness. This makes it hard to say if this failure is due to the model's inability to abstract the notion of curvature or just because of tuning errors on the first convolution layers. 

There is though evidence which supports the ability of the model to generalize:
- It succeeds on the second track after training with some data from it and running with the original speed of 9
- The outputs of the convolution layers show a clear focus on the lanes and on the curvature of the road

#### 3. Model parameter tuning

The model used an adam optimizer but the learning rate was tuned, especially while experimenting with different batch sizes.

#### 4. Appropriate training data

I used the following strategies to gather data:
- Drove as good as possible in the center of the lane
- Recorded small video chucks where I drove from a lane towards the center of the road
- Drove on the left and right edges of the road and then loaded those data sets with fixed angles navigating toward the center of the road
- Drove backwards

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Here is an outline of my approach:

1. MVP:
    - Generated some basic data driving in the middle of the road
    - Used a simple LeNet architecture
    - Fine-tuned the model till it 'made sense' and could drive the car in a rudimentary way
    - Added tools to visualize the convolution layers and check the distribution of the angle in the test dataset vs the distribution of the angle predictions in the test dataset
    - Set up a manual workflow to test the exported models in the simulator and identify the error cases
    - Implemented a random flip augmentation to easily remove the angle bias from the level

1. Upgraded to the nvidia publication architecture:
    - Read the publication
    - Developed the architecture in keras following some of the best practices related to CNNs in order to get the details right (e.g. use batch normalization instead of dropout)
    - Moved the project to an AWS instance with a GPU and setup scripts to setup the instance and move assets between the instance and my local workstation
    - Adjusted my pipeline to run with the new architecture
    - Tuned the model a bit to get better results
    
1. More data:
    - Drove more rounds on the center of the level
    - Drove backwards
    - Tried out several data augmentations like horizontal shift and color shifts
    - Tuned a bit the learning rate
    - Gathered data driving from the edges towards the center of the road
    - Gathered data driving on the sides of the road
    - Generated data on the second level

The final results were:
- The car could drive through the first level with the tweaked speed of 30.19. Look at `video_first_track_fast.mp4`.
- The car could drive through the first level with the original speed of 9. Look at `video_first_track.mp4`.
- The car could drive through the second level with the original speed of 9 and some manual interventions. Look at video `video_second_track.mp4`.
    
#### 2. Final Model Architecture

Here is a summary of the layers of the final architecture:

![alt Caption][model_layers]

#### 3. Creation of the Training Set & Training Process

I will analyze the strategies mentioned above.

1. Drove as good as possible in the center of the lane:
    - Gathered 6 datasets with ~3K images each: 'first_drives', '3_forward_rounds', '3_backward_rounds', '2_rounds_for_testing', '3_more_fw_rounds', '3_more_bw_rounds'
    - More data improved the performance of the model
    - Still the car could not make some of the sharper turns
    - When the car left the middle of the street it had very little knowledge on what to do
    
1. Recorded small video chucks where I drove from a lane towards the center of the road:
    - Gathered 3 datasets with ~3K images each: 'edges_forward', 'edges_backward', 'edges_smooth'
    - This enabled the car to succeed on the hard turns
    - Was relatively tedious as I had to start and stop the video all the time
    
1. Drove on the left and right edges of the road and then loaded those data sets with fixed angles navigating toward the center of the road:
    - Gathered 2 datasets: 'driving_on_the_edge_fw_left', 'driving_on_the_edge_fw_right'
    - Set respectively steering angle values of 0.15 and -0.15
    - Made the car more robust on turns
    - A bit of a hack since it would not generalize well on sharper corners
    - Would still trust this method more to ensure driving safety
- Drove backwards:
    - Reduced the bias towards the left oriented track
    - Generated more data
    
#### 4. Suggestions for improvement

Here are some ideas that I would follow if I were to continue the project:

- Gather more data by driving on separate parts of the road and replace the recorded angles with constant angles toward the center of the road
- Get more output from the simulator, e.g. distance from the center of the road, stepping on the boarders of the road, in order to create a better target metric. One example could be minimizing a penalty which consists of the distance from the center plus a higher penalty for crossing a line.
- Use multiple consecutive frames and build a time aware model. For example stack consecutive frames on a 4D tensor and apply 3D convolutions or use RNNs.