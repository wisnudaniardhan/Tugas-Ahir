import cv2

# Membaca gambar dalam format RGB
image = cv2.imread('wajah.jpg')

# Menerapkan conversi RGB ke HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Menampilkan hasil
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows