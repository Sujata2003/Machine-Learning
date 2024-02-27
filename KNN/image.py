# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:13:43 2024

@author: delll
"""

import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import blob_dog,blob_log,blob_doh
from math import sqrt
from skimage.color import rgb2gray
import glob
from skimage.io import imread
image = imread('C:/Users/delll/Pictures/sujata/amazing.jpg')
plt.imshow(image)
plt.show()


