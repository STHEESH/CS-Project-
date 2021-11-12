from PIL import Image
import numpy as np
import os, os.path, time
import matplotlib.image as img
import csv
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag


def save_in_csv():
    csv_path=''
    format='.png'
    myDir = "plates"
    def createFileList(myDir, format='.png'):
        fileList = []
        for root, dirs, files in os.walk(myDir, topdown=False):
                for name in files:
                   print(name)
                   if name.endswith(format):
                      fullName = os.path.join(root, name)
                      fileList.append(fullName)
        return fileList



    for image in createFileList(myDir):
        import numpy as np
        import matplotlib.image as img
        imageMat = img.imread(image)
        print("Image shape:", imageMat.shape)
        if len(imageMat.shape) == 2:
            x,y=imageMat.shape
            image_mat = imageMat.reshape(x,y,-1)
        else:
            image_mat=imageMat
 
# if image is colored (RGB)
        if(image_mat.shape[2] != -1):  
  # reshape it from 3D matrice to 2D matrice
            imageMat_reshape = image_mat.reshape(image_mat.shape[0],
                                      -1)
            print("Reshaping to 2D array:",
            imageMat_reshape.shape)
# if image is grayscale
        else:
            imageMat_reshape = image_mat
        np.savetxt(csv_path,
                imageMat_reshape)
    
        loaded_2D_mat = np.loadtxt(csv_path) 
# reshaping it to 3D matrice
        loaded_mat = loaded_2D_mat.reshape(loaded_2D_mat.shape[0],
                                       loaded_2D_mat.shape[1] // image_mat.shape[2],
                                       image_mat.shape[2])
 
        print("Image shape of loaded Image:",
          loaded_mat.shape)

    
