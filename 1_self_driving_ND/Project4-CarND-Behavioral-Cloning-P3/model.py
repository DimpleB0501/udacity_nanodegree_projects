import csv
import cv2
import numpy as np
from nvidiaCNN import model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self, model=None, epochs=5):
        self.data = []
        self.model = model
        self.epochs = epochs
        self.training_samples = []
        self.validation_samples = []
        self.base_path = './data'
        self.image_path = self.base_path + '/IMG/'
        self.driving_log_path = self.base_path + '/driving_log.csv'

    def import_data(self):
        with open(self.driving_log_path) as csvfile:
            reader = csv.reader(csvfile)
            firstLine = True
            for line in reader:
                if firstLine:
                    firstLine = False
                    continue 
                self.data.append(line)
        return None
    
    def generator(self, samples, batch_size=32):
        num_samples = len(samples)

        while True:
            shuffle(samples) #shuffling the images

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images, steering_angles = [], []

                for batch_sample in batch_samples:
                    for i in range(0,3): #3 images: center, left and right                        
                        name = batch_sample[i].split('/')[-1]
                        image = cv2.cvtColor(cv2.imread( self.image_path + name), cv2.COLOR_BGR2RGB) #to match drive.py 
                        angle = float(batch_sample[3]) #getting the steering angle measurement
                        images.append(image)
                        
                        # introducing correction for left and right images
                        # if image is in left, steering angle correction by +0.2
                        # if image is in right, steering angle correction by -0.2
                        
                        if(i==0):
                            steering_angles.append(angle)
                        elif(i==1):
                            steering_angles.append(angle+0.2)
                        elif(i==2):
                            steering_angles.append(angle-0.2)
                        
                        # Augmentation of data. Take images flip it and negate the measurement
                        images.append(cv2.flip(image,1))
                        if(i==0):
                            steering_angles.append(angle*-1)
                        elif(i==1):
                            steering_angles.append((angle+0.2)*-1)
                        elif(i==2):
                            steering_angles.append((angle-0.2)*-1)                            

                X_train, y_train = np.array(images), np.array(steering_angles)
                yield shuffle(X_train, y_train)    

    def run(self):
        #split data
        train_samples, validation_samples = train_test_split(self.data, test_size=0.2) #split data into 80:20 ratio
        
        # compile and train the model using the generator function
        train_generator = self.generator(train_samples, batch_size=32)
        validation_generator = self.generator(validation_samples, batch_size=32)
        self.model.fit_generator(generator=train_generator,
                                 validation_data=validation_generator,
                                 epochs=self.epochs,
                                 steps_per_epoch=len(train_samples),
                                 validation_steps=len(validation_samples))
        self.model.save('model.h5')

def main():
    # Instantiate the pipeline
    pipeline = Pipeline(model=model(), epochs=5)

    # Feed driving log data into the pipeline
    pipeline.import_data()
    # Start training
    pipeline.run()
    
    print ('Completed')

if __name__ == '__main__':
    main()
