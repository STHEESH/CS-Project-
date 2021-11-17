from matplotlib import image
from save_in_csv import predict_image
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from nembir_pilate_localijation import main as number_plate_localizer
from save_in_csv import *
from PIL import Image
from imaje_resije import conTO28x28, image_resize_sklearn


#********************************************************************************************************************************************************************************
def tk_app():    
    import tkinter as tk
    from tkinter import filedialog
    from tkinter.filedialog import askopenfile
    from PIL import Image, ImageTk

    my_w = tk.Tk()
    my_w.geometry("600x600")  # Size of the window 
    my_w.title('Automatic NumberPlate Recognition')
    my_w.config(bg = "BLACK")
    my_font1=('times', 24, 'bold')
    my_font2=('Times New Roman', 16, 'bold' )
    
    l1 = tk.Label(my_w,text='Automatic NumberPlate Recognition',width=32,font=my_font1)  
    l1.grid(row=1,column=1,columnspan=4)
    l2 = tk.Label(my_w,text='\n\n\n\nUpload Image Of Car Here',width=32,font=my_font2 ,fg = "WHITE" , background="BLACK")  
    l2.grid(row=4,column=1,columnspan=4)

    b1 = tk.Button(my_w, text=' Click Here to Upload File', 
    width=20,command = lambda:upload_file() , font=('Times New Roman', 8, 'bold' ))
    b1.grid(row=5,column=1,columnspan=4)

    def upload_file():
        f_types = [('Jpg Files', '*.jpg'),
        ('PNG Files','*.png')]   # type of files to select 
        filename = tk.filedialog.askopenfilename(filetypes=f_types)
                 
        number_plate_localizer(filename)
        format='.png'
        myDir = "plates"
        def createFileList(myDir, format='.png'):
            fileList = []
            for root, dirs, files in os.walk(myDir, topdown=False):
                for name in files:
                    # print(name)
                    if name.endswith(format):
                        fullName = os.path.join(root, name)
                        fileList.append(fullName)
            return fileList
        plates = createFileList(myDir)
        if plates == []:
            l3 = tk.Label(my_w,text='Number PLate not found',width=32,font=("Times New Roman ", 8 , 'bold') ,fg = "WHITE" , background="BLACK")  
            l3.grid(row=4,column=1,columnspan=4)
        else:
           # try:
                for image in createFileList(myDir):
                    pred = predict_image(image)
                    if len(pred) > 3:
                        img=Image.open(image)
                        img=ImageTk.PhotoImage(img) 
                        label=tk.Label(image = img)
                        label.grid(row=8 , column=1)
                        e2 =tk.Label(my_w , text=f"Prediction ,{pred}")
                        e2.grid(row = 10 , column = 1)
                        save_in_csv() 
           # except:
                e2 =tk.Label(my_w , text=f"Number PLate Not Found")
                e2.grid(row = 10 , column = 1)

    my_w.mainloop()   # Keep the window open

def stream_lit_app():
#STREAMLIT_APP
    try:
        import cv2
        from nembir_pilate_localijation import main as number_plate_localizer
        import streamlit as st  #Web App
        from PIL import Image #Image Processing
        import numpy as np #Image Processing 
        import os
        import easyocr
        
#title
        st.set_page_config(page_title="Autoatic NumberPLate Recogniton" )
        st.title("NUMBER PLATE RECOGNITION")
#subtitle
        st.markdown("")

#image uploader
        image = st.sidebar.file_uploader(label = "Upload the image of the car here",type=['png','jpg','jpeg'])
        if image is not None:
            #CHECKING IF IMAGE EXISTS OR NOT 

            input_image = Image.open(image) #read image
            file_details = image.name,image.type
            with open(os.path.join("tempdir" ,"temp.png"),"wb") as f:
                f.write(image.getbuffer())
            #saving numberplate in a directory called tempdir
            
            Type = st.radio("" , ["EasyOCR" , "ML Model"])

        
            st.sidebar.image(input_image) 
        #display image
            st.sidebar.write(file_details[0])
            
        #displaying name of file user uploaded
        
            st.sidebar.success("Image successfully uploaded!")
            st.balloons()
            if Type == "ML Model":
                if st.button("Click here to read the numer plate!"):
                    number_plate_localizer("tempdir/temp.png")
                    format='.png'
                    myDir = "plates"
                    def createFileList(myDir, format='.png'):
                        fileList = []
                        for root, dirs, files in os.walk(myDir, topdown=False):
                           for name in files:
                             # print(name)
                              if name.endswith(format):
                                 fullName = os.path.join(root, name)
                                 fileList.append(fullName)
                        return fileList
                    plates = createFileList(myDir)
                    if plates == []:
                        st.write("NumberPlate Not Found")
                    else:
                       try:
                        for image in createFileList(myDir):
                           pred = predict_image(image)
                           if len(pred) > 3:
                             img=Image.open(image)
                             st.image(img , caption="Localized Numberplate")
                     
                             st.write("prediction =",  pred)
                             save_in_csv()
                       
               
                       except:                
                        st.write("Can't Read The Numberplate")
            elif Type == "EasyOCR":
                if st.button('Click Here To Read The numberplate') :
                    number_plate_localizer("tempdir/temp.png")
                    format='.png'
                    myDir = "plates"
                    def createFileList(myDir, format='.png'):
                        fileList = []
                        for root, dirs, files in os.walk(myDir, topdown=False):
                           for name in files:
                             # print(name)
                              if name.endswith(format):
                                 fullName = os.path.join(root, name)
                                 fileList.append(fullName)
                        return fileList
                    plates = createFileList(myDir)
                    if plates == []:
                        st.write("NumberPlate Not Found")
                    else:
                       try:
                        directory = "datasets"
                        for image in createFileList(myDir):
                            Reader = easyocr.Reader(['en'] , model_storage_directory= directory)
                            text =Reader.readtext(image , paragraph= False)
                            text_ = ""
                            accuracy=0
                            for i in range(len(text)):
                            
                               text_+=text[i][1].rstrip("\n")
                               accuracy += float(text[i][2])
                            st.image(image)
                               
                            st.write( "Prediction : "  , text_)
                       except:
                            st.write("Number Plate Not Found")
                   

        else:
         st.sidebar.write("Upload an Image")

       
    except:
        #st.write("Can't Read the Numberplate")
        pass
       
                     

        st.caption('''BY Sathish 
                  and Amogh''')     

#*********************************************************************************************************************************************************************************


stream_lit_app()


            
          





