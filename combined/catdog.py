import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import sys

NUM_CLASSES = 2

def load_data(file_path):
    print(f'''\nLoading input data from "{file_path}"\n''')
    file = np.load(file_path)
    data_train = file['data_train'].astype('float32') / 255.
    label_train = keras.utils.to_categorical(file['label_train'], NUM_CLASSES)
    data_test = file['data_test'].astype('float32') / 255.
    label_test = keras.utils.to_categorical(file['label_test'], NUM_CLASSES)
    print("data_train: ", data_train.shape, data_train.dtype)
    print("data_test: ", data_test.shape, data_test.dtype)
    print("label_train:", label_train.shape, label_train.dtype)
    print("label_test:", label_test.shape, label_test.dtype)
    return data_train, label_train, data_test, label_test

def mlp_model(shape = (4096,)):
    model = model = Sequential()
    model.add(Dense(100, activation='sigmoid', input_shape = shape))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='mean_squared_error',
                optimizer=RMSprop(),
                metrics=['accuracy'])
    return model

def cnn_model(shape = (64,64,1)):
    model = model = Sequential()
    model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = shape))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64,(3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def train(model, epochs, batch_size, data_train, label_train, data_test, label_test):
    print('\nTraining new model\n')
    
    print('epochs: ', epochs)
    print('batch size: ', batch_size)

    print("data_train: ", data_train.shape, data_train.dtype)
    print("data_test: ", data_test.shape, data_test.dtype)
    print("label_train:", label_train.shape, label_train.dtype)
    print("label_test:", label_test.shape, label_test.dtype)

    model.summary()

    history = model.fit(data_train, label_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(data_test, label_test))
    
    score = model.evaluate(data_test, label_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return history, score

def graph_accuracy_loss(history, epochs, file_accuracy='model_acc.png', file_loss='model_loss.png'):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.axis([0.0, epochs, 0.0, 1.0])
    # plt.show()
    plt.savefig(file_accuracy, dpi=300)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.axis([0.0, epochs, 0.0, 1.0])
    # plt.show()
    plt.savefig(file_loss, dpi=300)

def run_mlp(epochs, batch_size, data_train, label_train, data_test, label_test):
    # input is (4096)
    model = mlp_model()
    history, score = train(model, epochs, batch_size, data_train, label_train, data_test, label_test)
    graph_accuracy_loss(history, epochs, 'mlp_accuracy.png', 'mlp_loss.png')

def run_cnn(epochs, batch_size, data_train, label_train, data_test, label_test):
    # input is (64,64,1)
    model = cnn_model()
    data_train = data_train.reshape(data_train.shape[0], 64, 64, 1)
    data_test = data_test.reshape(data_test.shape[0], 64, 64, 1)
    history, score = train(model, epochs, batch_size, data_train, label_train, data_test, label_test)
    graph_accuracy_loss(history, epochs, 'cnn_accuracy.png', 'cnn_loss.png')

def main(argv):
    if (len(argv) < 2):
        print('python catdog.py <path input data (an .npz)> <epochs = 20> <batch_size = 128>')
        return
    
    data_train, label_train, data_test, label_test = load_data(argv[1])
    
    epochs = 20
    if (len(argv) >= 3):
        epochs = int(argv[2])
    
    batch_size = 128
    if (len(argv) >= 4):
        batch_size = int(argv[3])
    
    run_mlp(epochs, batch_size, data_train, label_train, data_test, label_test)

    run_cnn(epochs, batch_size, data_train, label_train, data_test, label_test)

if __name__ == '__main__':
    main(sys.argv)