import tensorflow as tf
import keras_cv
import os
import tqdm
import cv2
import matplotlib
import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from keras_cv import visualization
from keras_cv import bounding_box

if __name__ == "__main__":

    #Setting capture device with USB camera using integer 1. Use 0 for built in cameras.
    cap = cv2.VideoCapture(1)

    #Downloading the pretrained computer vision model.
    pretrained_model = keras_cv.models.RetinaNet.from_preset(
        "retinanet_resnet50_pascalvoc",
        bounding_box_format="xywh")

    #Adjusting model input image dimensions
    inference_resizing = keras_cv.layers.Resizing(
        640,640,pad_to_aspect_ratio=True,bounding_box_format="xywh"
    )

    #Creating a mapping of potential labels
    label_names= [
        "Aeroplane",
        "Bicycle",
        "Bird",
        "Boat",
        "Bottle",
        "Bus",
        "Car",
        "Cat",
        "Chair",
        "Cow",
        "Dining Table",
        "Dog",
        "Horse",
        "Motorbike",
        "Person",
        "Plotted Plant",
        "Sheep",
        "Sofa",
        "Train",
        "Tvmonitor",
        "Total"
    ]
    id2label = {k:v for k, v in enumerate(label_names)}

    #Creates a continuous loop that is only interrupted when the key q is pressed. This allows for the video to be continuously captured and 
    #fed into the model to predict what the object is. Returning the predicted object while creating a label and bounding box for all detected 
    #objects in the frame.

    while True:
        ret, frame = cap.read()
        h, w , c = frame.shape
        batch = frame.reshape(1, h, w, c)
        image_batch=inference_resizing(batch)
        y_pred = pretrained_model.predict(image_batch)
        gallery = visualization.plot_bounding_box_gallery(
                    image_batch,
                    value_range=(0,255),
                    rows=1,
                    cols=1,
                    y_pred = y_pred,
                    scale=5,
                    font_scale=0.7,
                    bounding_box_format="xywh",
                    class_mapping=id2label,
                )
        fig = gallery.canvas.draw()
        buf = gallery.canvas.tostring_rgb()
        ncols, nrows = gallery.canvas.get_width_height()
        figure = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        cv2.imshow("image",figure)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()