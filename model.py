import os
import csv
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from PIL import Image

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda, Cropping2D, LSTM, Dropout, BatchNormalization, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import *

from process import * 

DATA_PATH = './data2/'
IMG_PATH  = DATA_PATH + 'IMG/'
np.random.seed(98886)

samples = []
with open(DATA_PATH +'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        l = [line[0],line[1],line[2], float(line[3])]
        
        #k = np.array([( line[0], line[1], line[2], float(line[3]) )], 
        #             dtype=[('center', '|S140'), ('left', '|S140'), ('right', '|S140'), ('angle', '>f4')])
 
        samples.append(l)

def sliding_mean(data_array, N=5):  
    data_array = np.asarray(data_array, dtype = float) 
    cumsum = np.cumsum(np.insert(data_array, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

#print (type(samples[:][3]))

def avg(samples):
    samples = np.array(samples)
    mean_angle = samples[:,3].astype(float)
    mean_angle = np.append(np.append(np.zeros((2)), sliding_mean(samples[:,3]) ) , np.zeros((2)) )
    samples[:,3] = mean_angle


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

BATCH_SIZE = 28

def generator(samples, batch_size=32):
    num_samples = len(samples)
    shuffle(samples)
    
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = process_file(IMG_PATH + batch_sample[0].split('/')[-1]);
                center_angle = float(batch_sample[3])
                
                left_image  = process_file(IMG_PATH + batch_sample[1].split('/')[-1]);
                right_image = process_file(IMG_PATH + batch_sample[2].split('/')[-1]);
                left_a = correction + center_angle
                right_a = -correction + center_angle 
                
                images.extend([center_image, np.fliplr(center_image), left_image, np.fliplr(left_image), right_image, np.fliplr(right_image)])
                angles.extend([center_angle, -center_angle, left_a, -left_a, right_a, -right_a ])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator      = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    
def nvidia():
    """
    NVIDIA Model
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    m = Sequential()
    #m.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape ))
    m.add(Lambda((lambda x: x/255.0 - 0.5), input_shape=input_shape ))
    m.add(BatchNormalization(epsilon=0.001))
    m.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    m.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    m.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    m.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    m.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    m.add(Flatten())
    #m.add(LSTM(1, stateful=True, return_sequences=False))
    m.add(Dense(1164, activation='relu'))
    m.add(Dense(100, activation='relu'))
    m.add(Dense(50, activation='relu'))
    m.add(Dense(10, activation='relu'))
    m.add(Dense(1, activation='tanh'))
    return m

filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
stop_early = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=5, verbose=0)


model = nvidia()
optimizer = Adam(lr=0.001)
model.compile(loss='mean_absolute_error', optimizer = optimizer)
model.summary()

model.fit_generator(train_generator, 
                    steps_per_epoch = len(train_samples) // BATCH_SIZE,
                    #samples_per_epoch = samples_per_epoch, 
                    validation_data   = validation_generator, 
                    validation_steps  = len(validation_samples) // BATCH_SIZE, 
                    epochs=50,
                    callbacks = [checkpoint, stop_early])