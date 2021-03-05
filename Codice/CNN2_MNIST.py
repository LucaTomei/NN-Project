import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras import backend as K

from keras.layers import LeakyReLU, PReLU
from keras_contrib.layers import SReLU

batch_size = 512
num_classes = 10
epochs = 12

img_chan, img_rows, img_cols = 1, 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 3

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=input_shape)
model.add(Activation('relu'))
#model.add(LeakyReLU())

model.add(MaxPooling2D((nb_pool, nb_pool)))
model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))
#model.add(LeakyReLU())

model.add(Dense(num_classes, activation='softmax'))

opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])