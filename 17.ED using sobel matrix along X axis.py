#17.Edge detection using Sobel Matrix along X axis
import cv2
import numpy as np

# Load the image
img = cv2.imread('import cv2
import numpy as np

# Read the input image
img = cv2.imread('input_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Laplacian filter with diagonal extension
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacian = cv2.filter2D(gray, -1, kernel)

# Add the Laplacian image to the original image to sharpen it
sharpened = cv2.add(gray, laplacian)

# Display the original and sharpened images
cv2.imshow('Original Image', gray)
cv2.imshow('Sharpened Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
', 0)

# Apply Sobel filter along the X-axis
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# Convert the output to absolute values
sobelx = np.abs(sobelx)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Sobel Edge Detection (X-axis)', sobelx) 
cv2.waitKey(0)
cv2.destroyAllWindows()
