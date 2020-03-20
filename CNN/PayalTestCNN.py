import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D , Flatten, BatchNormalization, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.utils import np_utils
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
    # print(images_scanned)
    categories = i.split(".")
    temp_img = cv2.imread(os.path.join(basePath, i))
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
train_imgs = np.array(train_imgs).reshape(-1, 64,64,3) # -1: We want numpy to figure it out; (64,64,1): 64X64 is image dimension and 1 is 1D(grey scale)
train_labels = np.array(train_labels)
train_imgs = train_imgs/255.0
#Preprocess test data"
test_imgs = np.array(test_imgs).reshape(-1, 64,64,3)
test_labels = np.array(test_labels)
test_imgs = test_imgs/255.0



model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = "relu", input_shape = (64,64,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (5,5), activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = (5,5), activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(train_imgs, train_labels, epochs=1, batch_size=200, validation_split=0.2)

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
plt.axis([0.0, 1 , 0.0, 1.0])
# plt.show()
plt.savefig("cnnAccuracy.png", dpi=300)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.axis([0.0, 1, 0.0, 1.0])
# plt.show()
plt.savefig("cnnLoss.png", dpi=300)



