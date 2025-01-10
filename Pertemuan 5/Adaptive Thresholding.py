import cv2

# Membaca gaambar dalam grayscale
image = cv2.imread('wajah.jpg', 0)

# Menerapkan adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Menmpilkan hasil 
cv2.imshow('Adaptive Thresholding', adaptive_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()