import sys

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import plot_model
import matplotlib.pyplot as plt

num_classes = 2


def get_data(file_path):
    # file = np.load('../Data/converted.npz')
    file = np.load(file_path)
    data_train = file['data_train']
    label_train = file['label_train']
    data_test = file['data_test']
    label_test = file['label_test']

    return data_train, label_train, data_test, label_test

def make_sequential(activation='sigmoid', hidden_shape=[100, 10], input_size=4096 ):

    model = Sequential()
    model.add(Dense(hidden_shape[0], activation=activation , input_shape = (input_size,)))
    for layer in hidden_shape[1:]:
        model.add(Dense(layer, activation=activation))

    model.add(Dense(num_classes, activation='softmax'))

    return model


def mlp_catdog(file_path = './Data/converted.npz', activ='sigmoid', batch_size = 128, epochs = 20, hidden_shape=[500, 100]):
    data_train, label_train, data_test, label_test = get_data(file_path)

    # print(label_train)

    data_train = data_train.astype('float32')
    data_test = data_test.astype('float32')

    data_train /= 255
    data_test /= 255

    print("data_train: ", data_train.shape, data_train.dtype)
    print("data_test: ", data_test.shape, data_test.dtype)

    # convert class vectors to binary class matrices
    label_train = keras.utils.to_categorical(label_train, num_classes)
    label_test = keras.utils.to_categorical(label_test, num_classes)

    print("label_train:", label_train.shape, label_train.dtype)
    print("label_test:", label_test.shape, label_test.dtype)

    model = make_sequential()
    
    model.summary()

    model.compile(loss='mean_squared_error',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    history = model.fit(data_train, label_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(data_test, label_test))
    score = model.evaluate(data_test, label_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # print(history.history)

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.axis([0.0, epochs, 0.0, 1.0])
    # plt.show()
    plt.savefig("model_acc.png", dpi=300)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.axis([0.0, epochs, 0.0, 1.0])
    # plt.show()
    plt.savefig("model_loss.png", dpi=300)





# mnist example pulled from Keras github to insure that it actually works
# https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
def mnist_example(batch_size = 128, num_classes = 10, epochs = 20):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def main(argv):
    # print(argv)
    # mnist_example()
    mlp_catdog(epochs=10)

if __name__ == '__main__':
    main(sys.argv)