import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD

from keras.layers import LeakyReLU, PReLU
from keras_contrib.layers import SReLU

batch_size = 32
num_classes = 10
epochs = 25

img_chan, img_rows, img_cols = 3, 32, 32
nb_filters = 32
nb_pool = 2
nb_conv = 3

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (nb_conv, nb_conv), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(32, (nb_conv, nb_conv), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D((nb_pool, nb_pool)))

model.add(Conv2D(64, (nb_conv, nb_conv), padding='same'))
model.add(Activation('relu'))


model.add(Conv2D(64, (nb_conv, nb_conv), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D((nb_pool, nb_pool)))

model.add(Conv2D(128, (nb_conv, nb_conv), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(128, (nb_conv, nb_conv), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D((nb_pool, nb_pool)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = SGD(lr=0.001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])