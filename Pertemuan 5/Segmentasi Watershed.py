import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('wajah.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# tresholding
ret, thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal 
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=2)

# Dilasi untuk menonjokan latar belakang
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Menghtung jakarak dari foreground
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg= cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255,0)

# Menandai area yang tidak jelas ( unkenow region)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Menandai komponen yang terhubung 
ret, markers = cv2.connectedComponents(sure_fg)

# Menambahkan satu untuk memastikan latar belakang tidak 0
markers = markers + 1
markers[unknown == 255] = 0

# menerapkan algoritma watershed
markers = cv2.watershed(image, markers)
image [markers == 1] = [255, 0, 0]

# menampilkan hasil 
cv2.imshow('Watershed Segmentation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()