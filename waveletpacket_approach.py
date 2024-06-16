import numpy as np
import pywt
import pywt.data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Feature extraction using waveletpacket transform.
img = mpimg.imread(r'liver-ultrasound-detection\train\train\images\149944.jpg')
B = np.mean(img, -1)
n = 2
w = 'db1'
coeffs = pywt.wavedec2(B, wavelet=w, level=n)
coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
    coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level]]
arr, coeff_slices = pywt.coeffs_to_array(coeffs)

imgr = pywt.idwt2(coeffs, 'db3', mode='periodization')
imgr = np.uint8(imgr)

