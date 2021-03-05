import keras
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.layers import LeakyReLU, PReLU, ELU
from keras_contrib.layers import SReLU

batch_size = 32
epochs = 100
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:])) 

#model.add(Activation('relu'))
model.add(LeakyReLU())

model.add(Conv2D(32, (3, 3)))

#model.add(Activation('relu'))
model.add(LeakyReLU())


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))

#model.add(Activation('relu'))
model.add(LeakyReLU())

model.add(Conv2D(64, (3, 3)))

#model.add(Activation('relu'))
model.add(LeakyReLU())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))

#model.add(Activation('relu'))
model.add(LeakyReLU())

model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])