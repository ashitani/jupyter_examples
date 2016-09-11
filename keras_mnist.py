import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

MAX_NUM=1000
X_train = X_train[:MAX_NUM,:,:].astype(np.float)/255.0
X_train = X_train.reshape(MAX_NUM,1,28,28)
y_train = y_train[:MAX_NUM]
y_train = np_utils.to_categorical(y_train, 10)

model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='same',input_shape=(1,28,28)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train, nb_epoch=20, batch_size=100,verbose=1)



