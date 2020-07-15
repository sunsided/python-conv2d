import cv2
import numpy as np

# load the image and scale to 0..1
image = cv2.imread('clock.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

# do-it-yourself convolution:
# For each pixel in the input image, we'll inspect its neighborhood. A 3x3 kernel thus peeks
# at every neighbor of a specific pixel (there are 8 pixel neighbors) whereas a 5x5 kernel
# peeks at two pixels in every direction (i.e. 24 pixel neighbors).

# A kernel of all ones is called a box blur and is simply averaging all neighbors (sum all, optionally divide by count).
kernel = (np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]))

# the weighed pixels have to be in range 0..1, so we divide by the sum of all kernel
# values afterwards
kernel_sum = kernel.sum()

# fetch the dimensions for iteration over the pixels and weights
i_width, i_height = image.shape[0], image.shape[1]
k_width, k_height = kernel.shape[0], kernel.shape[1]

# prepare the output array
filtered = np.zeros_like(image)

# Iterate over each (x, y) pixel in the image ...
for y in range(i_height):
    for x in range(i_width):
        weighted_pixel_sum = 0

        # Iterate over each weight at (kx, ky) in the kernel defined above ...
        # We interpret the kernel weights in a way that the 'central' weight is at (0, 0);
        # so the coordinates in the kernel are:
        #
        #  [ (-1,-1),  (0,-1),  (1,-1)
        #    (-1, 0),  (0, 0),  (1, 0)
        #    (-1, 1),  (0, 1),  (1, 1)
        #
        # This way, the pixel at image[y,x] is multiplied with the kernel[0,0]; analogous,
        # image[y-1,x] is multiplied with kernel[-1,0] etc.
        # The filtered pixel is then the sum of these, so that
        #
        #   weighted_pixel_sum = image[y-1,x-1] * kernel[-1,-1] +
        #                        image[y-1,x  ] * kernel[-1, 0] +
        #                        image[y-1,x+1] * kernel[-1, 1] +
        #                        image[y,  x-1] * kernel[ 0, 1] +
        #                        image[y,  x  ] * kernel[ 0, 0] +
        #                        etc.

        for ky in range(-(k_height // 2), k_height - 1):
            for kx in range(-(k_width // 2), k_width - 1):
                pixel = 0
                pixel_y = y - ky
                pixel_x = x - kx

                # boundary check: all values outside the image are treated as zero.
                # This is a definition and implementation dependent, it's not a property of the convolution itself.
                if (pixel_y >= 0) and (pixel_y < i_height) and (pixel_x >= 0) and (pixel_x < i_width):
                    pixel = image[pixel_y, pixel_x]

                # get the weight at the current kernel position
                # (also un-shift the kernel coordinates into the valid range for the array.)
                weight = kernel[ky + (k_height // 2), kx + (k_width // 2)]

                # weigh the pixel value and sum
                weighted_pixel_sum += pixel * weight

        # finally, the pixel at location (x,y) is the sum of the weighed neighborhood
        filtered[y, x] = weighted_pixel_sum / kernel_sum

cv2.imshow('DIY convolution', filtered)

# wait and quit
cv2.waitKey(0)
cv2.destroyAllWindows()
