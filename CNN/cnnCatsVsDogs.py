import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras import metrics
from keras import optimizers
from keras import losses
from keras import activations
import pickle

#pickle.dump( X, open( "train_x", "wb" ) )
#pickle.dump( y, open( "train_y", "wb" ) )

test_imgs = []
train_imgs = []
train_labels = []
basePath = "/Users/anantpathak/Learning/MachineLearning/catsvsdogs/train"
imgs = os.listdir("/Users/anantpathak/Learning/MachineLearning/catsvsdogs/train")
images_scanned = 0
for i in imgs:
    images_scanned = images_scanned+1
    print(images_scanned)
    categories = i.split(".")
    temp_img = cv2.imread(os.path.join(basePath, i), cv2.IMREAD_GRAYSCALE)
    temp_img = cv2.resize(temp_img, dsize=(64,64))
    if int(categories[1]) > int("10000"):
        test_imgs.append(temp_img)
        continue
    else:
        train_imgs.append(temp_img)
        if categories[0] == "cat":
            train_labels.append(0) #CAT  = 0
        else:
            train_labels.append(1) # DOG = 1
train_imgs = np.array(train_imgs).reshape(-1, 64,64,1)
train_labels = np.array(train_labels)
train_imgs = train_imgs/255.0

model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = train_imgs.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_imgs, train_labels, epochs=10, batch_size=32, validation_split=0.2)
model.summary()

predictions = model.predict(test_imgs)




