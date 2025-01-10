import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar 
image = cv2.imread('wajah.jpg', cv2.IMREAD_GRAYSCALE)

# Menampilkan gambar
cv2.imshow('Original Image', image)

# Menampilkan histogram
plt.hist(image.ravel(), bins=256, range=[0, 256])  
plt.title('Histogram of Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()  # Menampilkan histogram

# Menutup jendela
cv2.destroyAllWindows()
