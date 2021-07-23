import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv  
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.metrics import accuracy
import cv2 as cv
from PIL.Image import open
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test)=mnist.load_data()
x_train = x_train/255
x_test=x_test/255               #seprating image and labels       
                        #scaling , improves accuray 
x_train_flatten= x_train.reshape(len(x_train), 28*28)                      #makes it from 2d matrix to 1d
x_test_flatten= x_test.reshape(len(x_test), 28*28)


model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784,), activation = "sigmoid")
    
])

model.compile(
    optimizer = "adam",
    loss='sparse_categorical_crossentropy', 
    metrics = ["accuracy"]
    )

model.fit(x_train_flatten ,y_train,epochs=5 )         #epochs is no of iteration
                                                      #model is trained using model.fit


model.evaluate(x_test_flatten , y_test)


y_predicted = model.predict(x_test_flatten)
print(y_predicted[42])
print(np.argmax(y_predicted[42]))
plt.imshow(x_test[42])
plt.show()

'''
IMG_SIZE=28
img = input()
img = open(img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
resized = cv.resize(gray,(28,28),interpolation=cv.INTER_AREA)
newimg = tf.keras.utils.normalize(resized,axis=1)
newimg = np.array(newimg).reshape(-1,IMG_SIZE*IMG_SIZE,-1)
newimg = np.array(newimg).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print(newimg)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
resized = cv.resize(gray,(28,28),interpolation=cv.INTER_AREA)
newimg = tf.keras.utils.normalize(resized,axis=1)
newimg = np.array(newimg).reshape(-1,IMG_SIZE*IMG_SIZE,-1)
prediction = model.predict(newimg)
print(np.argmax(prediction))
'''