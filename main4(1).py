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
print(x_test_flatten.shape)
print(y_predicted[17])
print(np.argmax(y_predicted[17]))
plt.imshow(x_test[17])
plt.show()


from PIL import Image, ImageFilter


def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1]) 
    newImage = Image.new('L', (28, 28), (255))            # creates white canvas of 28x28 pixels

    if width > height:                                    # check which dimension is bigger
                                                          # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):                                #minimum is 1 pixel
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))              # paste resized image on white canvas
    else:
                                                    # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  
            nwidth = 1
                                                         # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))       # caculate vertical pozition
        newImage.paste(img, (wleft, 4))                 # paste resized image on white canvas

    

    ar = list(newImage.getdata()) 
    ar = [(255 - x) * 1.0 / 255.0 for x in ar]         #invert
    return ar

x=imageprepare("six.png")
x= np.reshape(x,10000,28*28) 
print(x.shape)

prediction = model.predict(x)
print(np.argmax(prediction))
