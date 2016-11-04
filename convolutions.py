import cv2
import numpy as np

# load the image and scale to 0..1
image = cv2.imread('clock.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

# load + show the original
cv2.imshow('original', image)

# horizontal edge detector
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('horizontal edges', filtered)

# vertical edge detector
kernel = np.array([[1,  1,  1],
                   [0,  0,  0],
                   [-1, -1, -1]])
filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('vertical edges', filtered)

# blurring ("box blur", because it's a box of ones)
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9.0
filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('blurred', filtered)

# sharpening
kernel = (np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]]))
filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('sharpened', filtered)

# wait and quit
cv2.waitKey(0)
cv2.destroyAllWindows()
