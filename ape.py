from re import L
from kivy.lang import builder
from matplotlib import image
from scipy.ndimage.measurements import label
from save_in_csv import predict_image
from tensorflow.keras.models import load_model
import numpy as np
from tkinter import *
from nembir_pilate_localijation import main as number_plate_localizer
from save_in_csv import *
from PIL import Image
from imaje_resije import conTO28x28, image_resize_sklearn


#********************************************************************************************************************************************************************************
def kivy_app():    
    from kivy.app import App
    from kivy.uix.gridlayout import GridLayout
    from kivy.lang import Builder
    from kivy.uix.label import Label
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.textinput import TextInput
    from kivy.core.window import Window
    Window.clearcolor = (0,0,0,0)
    
    Builder.load_string('''#:kivy 1.10.0
<MyWidget>:
    cols:2
    size: root.width , root.height 
    id:my_widget

   

    #FileChooserListView:
    FileChooserListView:
        id:filechooser
        on_selection:my_widget.selected(filechooser.selection)


    Image:
        id:image
        source:""''')

    class MyWidget(GridLayout): 
    
        def selected(self, filename):

            try:
                input_image = Image.open(filename[0]) 
                number_plate_localizer(filename[0])
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
                    l = Label( text = "NumberPlate Not Found")
                else:
                    try:
                        for image in createFileList(myDir):
                            pred = predict_image(image)
                            if len(pred) > 4:
                                img=Image.open(image)
                                l = Label(text="Localized Numberplate")
                                self.ids.image.source = image

                        
                                l_ = Label( text = f"prediction =  {pred}")
                                save_in_csv()
                    except:
                        pass

                
        
    
    
            except:
                pass
            
            

    
    
    
    class FileChooserWindow(App):
        
        def build(self):
    
            return MyWidget()
    
    
    
    
    if __name__ == "__main__":
        window = FileChooserWindow()
        window.run()

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
                           if len(pred) > 4:
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
                            
                            for i in range(len(text)):
                               text_+=text[i][1].rstrip("\n")
                            st.image(image)
                            if len(text_)> 4:
                                st.write( "Prediction : "  , text_)
                            else:
                                st.write("NumberPLate Not Found")
                       except:
                            #st.write("Number Plate Not Found")
                            pass
                   

        else:
         st.sidebar.write("Upload an Image")

       
    except:
        #st.write("Can't Read the Numberplate")
        pass
       
                     

        st.caption('''BY Sathish 
                  and Amogh''')     

#*********************************************************************************************************************************************************************************

kivy_app()
            
          





