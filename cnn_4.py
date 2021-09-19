import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from imaje_resije import *

mnist=tf.keras.datasets.mnist #Handwritten dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)
model= tf.keras.models.Sequential() 
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))#rectify linear unit
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3) 
#To train the model(epochs-->reptition of model)

loss,accuracy=model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

model.save('digits.model')
#To prevent training repeatedly
x = image_resize("eight.png")

img= cv.imread("eight.png")[:,:,0]
img=np.invert(np.array([img]))
prediction=model.predict(img)
print(prediction)
print('The result is probably:',np.argmax(prediction))#Index of highest value
