# Liver-Lesions-Detection-Ultrasound
## Objective
detect 7 classes of lesions in liver 
- 0	FFC
- 1	FFS
- 2	HCC
- 3	cyst
- 4	hemangioma
- 5	dysplastic
- 6	CCA

## Method
- base model: yolov9e object detection
- preprocessing 
  - equalization
  - smoothing filter
  - wavelet packet transform
  - Robert edge detection
  - Sobel and Laplacian

