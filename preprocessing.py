import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage import data, img_as_float
from skimage import exposure


img = mpimg.imread(r'test_machine\test_machine\2408.jpg')

# Equalization
def equalization(img):
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    # Equalization
    img_eq = exposure.equalize_hist(img)
    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_rescale, img_eq, img_adapteq

def GLT(image, transform, coeff = 0.5, gamma = 1.0):
    if transform == 'negative':
        table = np.array([256-1-i for i in np.arange(0,256)]).astype("uint8")
        
    elif transform == 'identity':
        table = np.array([i for i in np.arange(0,256)]).astype("uint8")
        
    elif transform == 'log':
        table = np.array([10*coeff*(np.log10(1+i)) for i in np.arange(0,256)]).astype("uint8")
        
    elif transform == 'invlog':
        table = np.array([10*coeff/(np.log10(1+i)+1) for i in np.arange(0,256)]).astype("uint8")
        
    elif transform == 'root':
        invGamma = 1.0/gamma
        table = np.array([coeff*((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
        
    elif transform == 'power':
        table = np.array([coeff*((i/255.0)**gamma)*255 for i in np.arange(0,256)]).astype("uint8")
        
    return cv2.LUT(image, table)

# lapacian + sobel

def lap_sobel(tran_power):
    laplacian = cv2.Laplacian(tran_power, ddepth = cv2.CV_8U, ksize = 11, scale = 5)
    sobelx = cv2.Sobel(tran_power, cv2.CV_8U, 1,0, ksize = 11)
    sobely = cv2.Sobel(tran_power, cv2.CV_8U, 0,1, ksize = 11)
    return laplacian, sobelx, sobely
