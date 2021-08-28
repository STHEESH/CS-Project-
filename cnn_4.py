import cv2 as cv #To read images
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from imaje_resije import *

'''
from PIl import Image
def imageParses(infile):
    outfile = os.path.splitext(infile)[0] + ".thumbnail"
    if infile != outfile:
        try:
            im = Image.open(infile)
            im = im.resize((28, 28))
            im.save(outfile, "JPEG")
        except IOError:
            print(f"cannot create thumbnail for {infile}")

imageParses('test.jpg')
'''
mnist=tf.keras.datasets.mnist #Handwritten dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#Split to trainig data and testing data

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

#1 input layer,2 hidden layers,1 output layer

#Normalise RGB and Grayscale to 0 and 1

model= tf.keras.models.Sequential() #Basic neural network
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#New layer(flatten for 1 dimensional)

model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))#rectify linear unit
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
#To connect the neural layers

model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))#Scales the probability
#To find the probabilty of a specific handwritten element

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#Optimizer-->Algorithm to change attributes of neural network. Adam uses a gradient
#Loss--> Prediction and calculation of error 


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
prediction=model.predict(img)#Gives softmax result(highest area where neuron is activated)
print(prediction)
print('The result is probably:',np.argmax(prediction))#Index of highest value
