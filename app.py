# Libraries
import tkinter as tk
import numpy as np
import tensorflow as tf
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import cv2
from keras.models import load_model

from src.components.data_transformation import DataTransformation

# Load your model
model = load_model("artifacts/Saved_Models/VGGNet")  # Path to your model

# Dictionary to label all traffic signs class.
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

# Initialise GUI
top = tk.Tk()
# Window dimensions (800x600)
top.geometry('800x600')
# Window title
top.title('Traffic sign classification')
# Window background color
top.configure(background='#CDCDCD')
# Window label
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
# Sign image
sign_image = Label(top)

# Function to resize and preprocess the input image (assuming grayscale)


def preprocess_image(image):
    resized_image = cv2.resize(image, (32, 32))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    gray_images = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    image = np.divide(gray_images, 255)

    return image


# Function to classify image
def classify(file_path):
    # Open the image file path
    image = cv2.imread(file_path)
    # Preprocess the test image
    preprocessed_image = preprocess_image(image)
    # Make predictions
    predictions = model.predict(np.array([preprocessed_image]))

    pred = np.argmax(predictions)
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#011638', text=sign)

# Function to show the "classify" button


def show_classify_button(file_path):
    # Create the button
    classify_b = Button(top, text="Classify Image",
                        command=lambda: classify(file_path), padx=10, pady=5)
    # Configure button colors
    classify_b.configure(background='#364156',
                         foreground='white', font=('arial', 10, 'bold'))
    # Configure button place (location)
    classify_b.place(relx=0.79, rely=0.46)

# Function to upload image


def upload_image():
    try:
        # Path of the image
        file_path = filedialog.askopenfilename()
        # Open file path
        uploaded = Image.open(file_path)
        uploaded.thumbnail(
            ((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


# Create "Upload" button
upload = Button(top, text="Upload an image",
                command=upload_image, padx=10, pady=5)
# "Upload" button colors and font
upload.configure(background='#364156', foreground='white',
                 font=('arial', 10, 'bold'))
# Button location
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
# Window title text
heading = Label(top, text="Know Your Traffic Sign",
                pady=20, font=('arial', 20, 'bold'))
# Window colors
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
