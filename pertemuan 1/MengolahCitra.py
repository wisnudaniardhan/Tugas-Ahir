import cv2

# Mwmbaca gambar
image = cv2.imread('wajah.jpg')

# menampilkan gambar 
cv2.imshow('Display window', image)

# menunggu hingga ada input dari keyboard
cv2.waitKey(0)

# menutup semua jendela
cv2.destroyAllWindows()