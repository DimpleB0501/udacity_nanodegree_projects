# Facial Keypoint Detection

## Project Overview

In this project, we build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. 

Following implementation looks at any image, detect faces (using haar cascades), and predicts the locations of facial keypoints on each face; examples of these keypoints are displayed below
![Obamas][./images/obamas.jpg]
![Facial Keypoint Detection][./images/output_keypt_detector.png]

The project will be broken up into a few main parts in four Python notebooks, 

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses



## Project 

### Local Environment Instructions
1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/P1_Facial_Keypoints.git
cd P1_Facial_Keypoints
```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.
Anaconda [installation](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart) and [installing](https://medium.com/analytics-vidhya/how-i-installed-cuda-10-0-for-pytorch-in-linux-mint-2ce26dd1930f) cuda 10.1 and cudnn for in-built Nvidia GPU. 
Follow the instructions to setup the enviornment in local GPU:
	- __Linux__ : 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
	pip3 install -r requirements.txt
	```

	
### Notebooks

1. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd
cd P1_Facial_Keypoints
```

2. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```
and run the notebooks.

### Evaluation
I implemented architecture described in [paper](https://arxiv.org/pdf/1710.00977.pdf) for this project. I removed the dropout layers at each layer and implemented the dropout only at fc2 layer. The data transforms implemented are rescaling/cropping, normalization, and turning input images into torch Tensors. As described in the paper I implemented Adam optimizer and mse loss function. The training batch size was 10 images and number of epochs was 25. 

