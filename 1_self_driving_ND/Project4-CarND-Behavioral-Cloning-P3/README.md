**Behavioral Cloning Project**

### Introduction:
Following writeup is an accompaniment to the submitted project 4 of Udacity's Self-Driving nanodegree program. In this project our goal is to train a network using Keras which can successfully emulate human driving behavior autonomously in a simulated enviornment.
We had an option to generate data using the simulator or use Udacity's sample dataset. I initially tried generating the data however I wasn't able to navigate through the simulation enviornment properly as I would mostly drive off the roads. Hence I chose to proceed with the dataset provided by Udacity.
Following the Udacity lectures I built the CNN based on NVIDIA's [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) paper. To avoid overfitting I have added a dropout layer after the first fully connected layer. The training and testing was done in Udacity' workspace.

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* nvidiaCNN.py contains the model architecture.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

### Model Architecture and Training Strategy
### Problem statement
We have a regression problem where our inputs are images from cameras in the car and we need to predict steering angles to stay on course of the road.

### Udacity dataset
We feed the data to model, this data is fed in the form of images captured by 3 dashboard cams: center, left and right. The output data contains a file data.csv which has the mappings of center, left and right images and the corresponding steering angle, throttle, brake and speed.

### Data
Using OpenCV we load the images. The images are converted from default (BGR) to RGB format to match drive.py. The size of the image is set to (160,320,3).

### Data augmentation
The images are shuffled. As suggested in the course I introduced a steering angle correction factor of 0.2 for left and right images, as the steering angle is associated with center images. For the left camera images I increased the steering angle by 0.2 and for right images I decreased it by 0.2. Also as mentioned in the course I flipped the center, left and right images using **cv2.flip** function, and adjusted their steering angle by multiplying the angles and their respective added corrections by -1.

### Data Preprocessing
I then preprocessed the data obtained by the above augmentation procedure by using lambda layer to normalize input images avoiding saturation and improving gradients preformance. The images from the simulator have a lot of useless information from the enviornment like trees, mountains, sky and the car's front. To avoid distracting the model I have cropped the images using keras **Cropping2D(cropping=((70,25),(0,0)))** layer  throwing away above portion of the images.

### Splitting in train and validation set
I have split the data into 80 percent training set and 20 percent validation set using **train_test_split** function from **sklearn.model_selection**.

### Data Generator
To use the entire RGB dataset for training would consume a large amount of memory. As suggested in the course I incorporated python generator so that a single batch of data (batch size = 32) is feed to the network and contained in the memory at a time.  

### Model architecture Design
The Nvidia architecture has 9 layers. NVIDIA models states that the convolution layers are meant to handle feature engineering and the fully connected layer is meant for predicting the steering angle. As an input the NVIDIA model takes image of shape (60,266,3) but following the course I have set the image  size of (160,320,3). I have left rest of the architecture same except after the first fully connected layer I have incorporated a drop out layer (rate:0.25) to combact overfitting. This is done so that the model can generalize on a track it has not seen.
Reading literature on activation units and why ELU activation were used against RELU, I came to know that they have marginally faster performance and lower loss as compared to RELU's, like RELU's they tackle the vanishing gradient problem and they improve the efficiency of gradient descent as they have negative values allowing them to push the mean activation closer to zero.

Following is the visualization of the layers used and its sizes

| Layer         		| Description    	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB Image                 	   		|
| Normalization     		| Normalize batch	                            |
| Cropping layer		| 65x320x3
| Convolution 5x5   | 2x2 stride, outputs 31x158x24 	|
| ELU activation		|												|
| Convolution 5x5	  | 2x2 stride, outputs 14x77x36   |
| ELU activation    |                                               |
| Convolution 5x5	  | 2x2 stride, outputs 5x37x48    |
| ELU activation    |                                               |
| Convolution 3x3	  | 1x1 stride, outputs 3x35x64    |
| ELU activation    |                                               |
| Convolution 3x3	  | 1x1 stride, outputs 1x33x64    |
| ELU activation    |                                               |
| Flatten           |                                               |
| Fully connected		| 2112 input, 100 output     					|
| ELU activation		|												|
| Dropout           | 0.25 keep probablility (training)              |
| Fully connected		| 100 input, 50 output     				     	|
| ELU activation		|												|
| Fully connected		| 50 input, 10 output     				     	|
| ELU activation		|												|
| Output         		| 10 input, 1 output     				     	|

As mentioned previously, the training took place on a Amazon EC2 instance. The network was built with Keras and used Tensorflow as the backend. The model architecture along with the data preprocessing (normalization and cropping layers) is implemented in **nvidiaCNN.py** file.


### Model Parameters
* No of epochs= 5
* Optimizer Used- Adam
* Learning Rate- Default 0.001
* Loss Function Used- MSE(Mean Squared Error as it is efficient for regression problem).

### Thought process
I followed Udacity's lectures to setup the data preprocessing, augmentation and architecture design blocks. To quickly test out the implemented code I trained on 1 epoch (loss:0.0303) and the steps per epoch was  minimal and set to  (len(train_samples) * 0.01). When I ran the model in the simulator the car was able to stay on track/ center of the road for the whole lap, however it drived slowly throughout the lap. This model is saved as 'model_backup.h5'. As suggested in the class 2-5 epochs should be sufficient for training the model, I increased the number of epochs to 5 with the intent to lower the MSE loss and ran the models through length of training samples as steps per epoch. and ensuring that the vehicle could stay on the track. The loss came down to 0.0062 at end of 1st epoch and the validation loss was 0.0216. However with the batch size of 128 it took more than 2.5 hrs to train one epoch. Since I didn't want to waste GPU time I reduce the batch size to 32.  I trained the network again at the end of the 5th epoch I got an trainning loss of 0.0015 and validation loss of 0.0169. Following snapshot displays the training and the validation losses at each epoch.
![Losses at each epoch](https://github.com/DimpleB0501/selfDrivingNanodegree/blob/master/Project4-CarND-Behavioral-Cloning-P3/images/Losses.png) 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
Following is the **output video** of the car autonomously driving through the lap in the simulator.

|Output video of car driving autonomously in simulator|
|:------------:|
|![Output video](https://github.com/DimpleB0501/selfDrivingNanodegree/blob/master/Project4-CarND-Behavioral-Cloning-P3/images/simulator.gif)|
|[Youtube Link](https://youtu.be/BZA1jUqT58Q)|

The model can be tested by running it through the simulator by executing
```sh
python drive.py model.h5
```

### Discussion
Overall, it was amazing to learn keras and how with few lines of code the complete architecture can be implemented.
In terms of the project. Having a chance to create my own data set was great. However I wasn't able to sucessfully do it. Due to time crunch I followed data augmentation as instructed in Udacity's lectures. It would have been great to implement better augmentation as taught in previous lectures. Also I am not quite sure that the trained model will run properly on a real enviorment as well as it hardly happens that vehicle runs in isolation on roads.
I also couldn't understand why a very long time is taken to obtain the **model.h5** file. It would have been great if heads up was provided initally regarding this, thus avoiding wasting of GPU resources and time.
