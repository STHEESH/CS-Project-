import tensorflow as tf

#MNIST
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train.shape

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[0] , cmap = plt.cm.binary)

#NORMALIZATION
x_train = tf.keras.utils.normalize (x_train, axis=1)
x_test = tf.keras.utils.normalize (x_test, axis=1)
plt.imshow(x_train[0] , cmap = plt.cm.binary)

import numpy as np
IMG_SIZE=28
x_trainr = np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
x_testr = np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)

#NEURAL NETWORK
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation('softmax'))

#TRAINING
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(x_trainr,y_train,epochs=5,validation_split=0.3)
test_loss,test_acc = model.evaluate(x_testr,y_test)

#PROCESSING IMAGE
import cv2
src = r'C:\Users\sathi\Desktop\EVERYTHING\CS PROJECT\six.png'
img = cv2.imread(src)
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)
newimg = tf.keras.utils.normalize(resized,axis=1)
newimg = np.array(newimg).reshape(-1,IMG_SIZE,IMG_SIZE,1)

#PREDICTING
prediction = model.predict(newimg)
print(np.argmax(prediction))