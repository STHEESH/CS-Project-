from PIL import Image, ImageFilter

def imageprepare(image):
    im = Image.open(image).convert('L')
    width = float(im.size[0])
    height = float(im.size[1]) 
    newImage = Image.new('L', (28, 28), (255))            # creates white canvas of 28x28 pixels

    if width > height:                                    # check which dimension is bigger
                                                          # Width is bigger. Width becomes 20 pixels.
        new_height = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (new_height == 0):                                #minimum is 1 pixel
            new_height = 1
        img = im.resize((20, new_height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - new_height) / 2), 0))  # calculate horizontal position
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

    

    img_ = list(newImage.getdata()) 
    img_ = [(255 - x) * 1.0 / 255.0 for x in img_]         #invert
    return img_






