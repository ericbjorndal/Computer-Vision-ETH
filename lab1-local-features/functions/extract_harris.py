import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    kernel = np.array([[-1/2, 0, 1/2]])
    Ix = signal.convolve2d(img, kernel, mode='same', boundary='symm', fillvalue=0)
    Iy = signal.convolve2d(img, np.transpose(kernel), mode='same', boundary='symm', fillvalue=0)
 
    # 2. (Optional) Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    
    #Ix = cv2.GaussianBlur(Ix, (0,0), sigma, borderType=cv2.BORDER_REPLICATE)
    #Iy = cv2.GaussianBlur(Iy, (0,0), sigma, borderType=cv2.BORDER_REPLICATE)

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur or scipy.signal.convolve2d to perform the weighted sum
    
    Mxx = cv2.GaussianBlur(Ix**2, (0,0), sigma, borderType=cv2.BORDER_REPLICATE)
    Mxy = cv2.GaussianBlur(Ix*Iy, (0,0), sigma, borderType=cv2.BORDER_REPLICATE)
    Myy = cv2.GaussianBlur(Iy**2, (0,0), sigma, borderType=cv2.BORDER_REPLICATE)

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    
    detM = Mxx * Myy - Mxy**2 
    traceM = Mxx + Myy
    C = detM - k * (traceM**2)
    
    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    
    local_max = ndimage.maximum_filter(C, size=3)
    y, x = np.where(np.logical_and(local_max == C, local_max > thresh))
    corners = np.stack((x,y), axis=1)

    return corners, C