import cv2 
import numpy as np

# Baca citra 
image = cv2.imread('wajah.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Lakukan thresholding untuk menghilangkan noise
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# morpological operator untuk memisahkan background dan foreground
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(thresh, kernel, iterations=3)

# Distance transfor untuk mendaptkan foreground
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# watershed
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Tandai komponen 
ret, markers = cv2.connectedComponents(sure_fg)

# tambahkan 1 pada marker sehingga background menjadi 1
markers = markers + 1
markers[unknown == 255] = 0

# terapkan watershed
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]

# tampilkan hasil
cv2.imshow('Watershed Segmentation', image)
cv2.waitKey(0)
cv2.destroyAllWindows
