from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Convolution2D
from keras.layers import Lambda, Cropping2D

def model():
    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # Crop image to only see section with road
    model.add(Cropping2D(cropping=((70,25),(0,0))))           

    #layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
    model.add(Convolution2D(24,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
    model.add(Convolution2D(36,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
    model.add(Convolution2D(48,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Convolution2D(64,3,3))
    model.add(Activation('elu'))

    #layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Convolution2D(64,3,3))
    model.add(Activation('elu'))

    #flatten image
    model.add(Flatten())

    #layer 6- fully connected 
    model.add(Dense(100))
    model.add(Activation('elu'))

    #Adding dropout layer to avoid overfitting. 
    model.add(Dropout(0.25))

    #layer 7- fully connected 
    model.add(Dense(50))
    model.add(Activation('elu'))


    #layer 8- fully connected 
    model.add(Dense(10))
    model.add(Activation('elu'))

    #layer 9- fully connected 
    model.add(Dense(1)) #layer thats predicts the steering angle
    
    model.compile(loss='mse',optimizer='adam')

    return model
