# CarND-Extended-Kalman-Filter-P1
Udacity Self-Driving Car Nanodegree - Extended Kalman Filter Implementation

# Overview
This project consists of implementing an [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter) with C++. A simulator provided by Udacity generates noisy RADAR and LIDAR measurements of the position and velocity of an object, and the Extended Kalman Filter code fuses those measurements to predict the position of the object. The communication between the simulator and the EKF is done using [WebSocket](https://en.wikipedia.org/wiki/WebSocket) using the [uWebSockets](https://github.com/uNetworking/uWebSockets) implementation on the EKF side.
Udacity provided all the framework required to implement EKF in C++. A LIDAR measures the object's x and y position pretty well and a RADAR gives an estimate of object velocities in the x and y direction (however measurements are noisy). Kalman filter algorithm can be organized into following steps:
 1. Initialize state and covariance matrices
 2. Make a prediction of a state based on previous sensor values and model.
 3. Updation step where the predicted and measured location are combined to give an updated location.
The process repeats as the object gets another sensor measurements. And as explained in the lectures the only difference between RADAR and LIDAR logic is the updation step.

# Instructions to run this project
- Clone the repo
- In terminal type `bash install-linux.sh` to install the necessary library and dependencies
- Create the build directory: `mkdir build`
- Move to the **build** directory: `cd build`
- Compile: `cmake .. && make`
- Run from **build** directory: `./ExtendedKF`
- With GPU enabled start the simulator. The output should be
	```
	Listening to port 4567
	Connected!!!
	```

# Results
The simulator connects right away. The simulator has 2 datasets. The simulator displays residual mean squared error (RMSE) in x-position, y-position, velocity in x-direction `VX` and velocity in y-direction `VY`. The aim of the project is to minimize RMSE.

Following displays the final state of the car after running the extendend kalman filter code on `dataset 1`:
![Dataset1-RMSE](https://github.com/DimpleB0501/selfDrivingNanodegree/blob/master/Project5-CarND-Extended-Kalman-Filter-Project/images/EKFDataSet1.png)
RMSE = [0.0973, 0.0855, 0.4513, 0.4399]

Following displays the final state of the car after running the extendend kalman filter code on `dataset 2`:
![Dataset2-RMSE](https://github.com/DimpleB0501/selfDrivingNanodegree/blob/master/Project5-CarND-Extended-Kalman-Filter-Project/images/EKFDataset2.png)
RMSE = [0.0726, 0.0965, 0.4216, 0.4932]
