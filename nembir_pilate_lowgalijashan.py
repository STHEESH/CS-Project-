import cv2 as cv
import numpy as np

'''****************************************************************************************'''
# IMAGE_ENHANCING_FUNCTIONS
def D_filter(image):
    return cv.filter2D(image , -1 , np.ones((5,5), np.float32)/25)


def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def noiseRemoval(image):
    return cv.bilateralFilter(image, 11, 17, 17)

def histogramEqualization(image):
    return cv.absdiffequalizeHist(image)

def morphologicalOpening(image, structElem):
    return cv.morphologyEx(image, cv.MORPH_OPEN, structElem, iterations=15)

def subtractOpenFromHistEq(histEqImage, morphImage):
    return cv.subtract(histEqImage, morphImage)

def tresholding(image):
    x,t=cv.threshold(image, 127, 255, cv.THRESH_BINARY, cv.THRESH_OTSU)
    return t
'''*********************************************************************************************************'''
# Edge Detection
def edgeDetection(image, threshold1, threshold2):
    cannyImage = cv.Canny(image, threshold1, threshold2)
    cannyImage = cv.convertScaleAbs(cannyImage)
    return cannyImage


def imageDilation(image, structElem):
    return cv.dilate(image, structElem, iterations=1)

#Contours
def findContours(image):
    newImage, contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ''' For this problem, number plate should have contours with a small area as compared to other contours.
        Hence, we sort the contours on the basis of contour area and take the least 10 contours'''
    return sorted(contours, key=cv.contourArea, reverse=True)[:10]

#Ramer-Douglas-Peucker Algorithm
def approximateContours(contours):
    approximatedPolygon = None
    for contour in contours:
        contourPerimeter = cv.arcLength(contour, True)
        approximatedPolygon = cv.approxPolyDP(contour, 0.06*contourPerimeter, closed=True)
        if(len(approximatedPolygon) == 4):                              #number plate is usually a rectangle
            break                                                       #therefore breaks when a quad is encountered
    return approximatedPolygon


# Highlighting the image
def drawLocalizedPlate(image, approximatedPolygon):
    M=cv.moments(approximatedPolygon)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    
    finalImage = cv.drawContours(image, [approximatedPolygon], -1, (0, 255, 0), 3)
    
    cv.circle(finalImage, (cX, cY), 7, (0, 255, 0), -1)
    cv.putText(finalImage, "Centroid of Plate: ("+str(cX)+", "+str(cY)+")", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return finalImage


'''***********************************************************************************************************'''







