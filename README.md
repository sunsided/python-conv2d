# 2D Convolutions in Python (OpenCV 2, numpy)

In order to demonstrate 2D kernel-based filtering without relying on library code too much, `convolutions.py` gives some examples to play around with.

```python
image = cv2.imread('clock.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('vertical edges', filtered)
```

In addition, `convolution_manual.py` implements a manual 2D convolution to explain the concept.