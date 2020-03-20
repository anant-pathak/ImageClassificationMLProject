import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
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
test_labels = []

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
    if int(categories[1]) > int("11000"):
        test_imgs.append(temp_img)
        if categories[0] == "cat":
            test_labels.append(0) #CAT  = 0
        else:
            test_labels.append(1) # DOG = 1
        continue
    else:
        train_imgs.append(temp_img)
        if categories[0] == "cat":
            train_labels.append(0) #CAT  = 0
        else:
            train_labels.append(1) # DOG = 1
#Preprocessing on Train data
train_imgs = np.array(train_imgs).reshape(-1, 64,64,1) # -1: We want numpy to figure it out; (64,64,1): 64X64 is image dimension and 1 is 1D(grey scale)
train_labels = np.array(train_labels)
train_imgs = train_imgs/255.0
#Preprocess test data"
test_imgs = np.array(test_imgs).reshape(-1, 64,64,1)
test_labels = np.array(test_labels)
test_imgs = test_imgs/255.0



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
              loss="mean_squared_error", # mean_squared_error | binary_crossentropy
              metrics=['accuracy'])

history = model.fit(train_imgs, train_labels, epochs=20, batch_size=128, validation_data=(test_imgs, test_labels)) #validation_split = 0.2
model.summary()
print("history keys = ", history.history.keys())
#storing the histor

# with open("History", 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
#predictions = model.predict(test_imgs)
# predicted_val = [int(round(p[0])) for p in predictions]
# submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})
# submission_df.to_csv("submission.csv", index=False)


#Graph plotting:

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot()
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.axis([0.0, 20 , 0.0, 1.0])
# plt.show()
plt.savefig("cnnAccuracy.png", dpi=300)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MSE Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.axis([0.0, 20, 0.0, 1.0])
# plt.show()
plt.savefig("cnnLoss.png", dpi=300)