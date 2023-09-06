# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:23:16 2022

@author: rajpr


"""
import scipy.io


mat = scipy.io.loadmat('SAl_KM_16_run1.mat');

from matplotlib import pyplot as plt

from skimage.color import label2rgb
plt.imshow(label2rgb(mat))
plt.show()