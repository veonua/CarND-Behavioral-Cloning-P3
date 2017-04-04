import cv2
import numpy as np
import matplotlib.image as mpimg

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

input_shape=(90,320,3) #(45,160, 3)

def process_image(arr):
    arr = arr[50:-20,:]
    yuv = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    #yuv = cv2.resize(, (160, 45) )
    #yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    
    return yuv

def process_file(path):
    return process_image( np.asarray(mpimg.imread(path)) )
