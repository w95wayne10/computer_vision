import cv2

lena_gray = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
upside_down_lena = lena_gray[::-1, :]
rightside_left_lena = lena_gray[:, ::-1]
diagonally_flip_lena = lena_gray.T

cv2.imwrite('upside_down_lena.bmp', upside_down_lena)
cv2.imwrite('rightside_left_lena.bmp', rightside_left_lena)
cv2.imwrite('diagonally_flip_lena.bmp', diagonally_flip_lena)