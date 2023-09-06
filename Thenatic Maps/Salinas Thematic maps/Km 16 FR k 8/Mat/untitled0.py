# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 00:09:38 2022

@author: rajpr
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image

#reading v 7.3 mat file in python

filepath = 'SAl_KM_16_FR_8_run8.mat';
f = h5py.File(filepath, 'r') #Open mat file for reading

#In MATLAB the data is arranged as follows:
#cjdata is a MATLAB struct
#cjdata.image is a matrix of type int16

#Before update: read only image data.   
####################################################################
#Read cjdata struct, get image member and convert numpy ndarray of type float
#image = np.array(f['cjdata'].get('image')).astype(np.float64) #In MATLAB: image = cjdata.image
#f.close()
####################################################################

#Update: Read all elements of cjdata struct
####################################################################
#Read cjdata struct
cjdata = f['cjdata'] #<HDF5 group "/cjdata" (5 members)>

# In MATLAB cjdata = 
# struct with fields:
#   label: 1
#   PID: '100360'
#   image: [512×512 int16]
#   tumorBorder: [38×1 double]
#   tumorMask: [512×512 logical]

#get image member and convert numpy ndarray of type float
image = np.array(cjdata.get('image')).astype(np.float64) #In MATLAB: image = cjdata.image

label = cjdata.get('label')[0,0] #Use [0,0] indexing in order to convert lable to scalar

PID = cjdata.get('PID') # <HDF5 dataset "PID": shape (6, 1), type "<u2">
PID = ''.join(chr(c) for c in PID) #Convert to string https://stackoverflow.com/questions/12036304/loading-hdf5-matlab-strings-into-python

tumorBorder = np.array(cjdata.get('tumorBorder'))[0] #Use [0] indexing - convert from 2D array to 1D array.

tumorMask = np.array(cjdata.get('tumorMask'))

f.close()
####################################################################

#Convert image to uint8 (before saving as jpeg - jpeg doesn't support int16 format).
#Use simple linear conversion: subtract minimum, and divide by range.
#Note: the conversion is not optimal - you should find a better way.
#Multiply by 255 to set values in uint8 range [0, 255], and covert to type uint8.
hi = np.max(image)
lo = np.min(image)
image = (((image - lo)/(hi-lo))*255).astype(np.uint8)

#Save as jpeg
#https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
im = Image.fromarray(image)
im.save("1.jpg")

#Display image for testing
imgplot = plt.imshow(image)
plt.show()