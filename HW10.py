import cv2
import numpy as np

LoG1 = [0,0,0,-1,-1,-2,-1,-1,0,0,0]
LoG2 = [0,0,-2,-4,-8,-9,-8,-4,-2,0,0]
LoG3 = [0,-2,-7,-15,-22,-23,-22,-15,-7,-2,0]
LoG4 = [-1,-4,-15,-24,-14,-1,-14,-24,-15,-4,-1]
LoG5 = [-1,-8,-22,-14,52,103,52,-14,-22,-8,-1]
LoG6 = [-2,-9,-23,-1,103,178,103,-1,-23,-9,-2]
LoG_mtx = np.array([LoG1,LoG2,LoG3,LoG4,LoG5,LoG6,LoG5,LoG4,LoG3,LoG2,LoG1])

DoG1 = [-1,-3,-4,-6,-7,-8,-7,-6,-4,-3,-1]
DoG2 = [-3,-5,-8,-11,-13,-13,-13,-11,-8,-5,-3]
DoG3 = [-4,-8,-12,-16,-17,-17,-17,-16,-12,-8,-4]
DoG4 = [-6,-11,-16,-16,0,15,0,-16,-16,-11,-6]
DoG5 = [-7,-13,-17,0,85,160,85,0,-17,-13,-7]
DoG6 = [-8,-13,-17,15,160,283,160,15,-17,-13,-8]
DoG_mtx = np.array([DoG1,DoG2,DoG3,DoG4,DoG5,DoG6,DoG5,DoG4,DoG3,DoG2,DoG1])

def conv(img_org, patn):
  img = img_org.astype(float)
  m,n = patn.shape
  M,N = img.shape
  result = np.zeros((M-m+1,N-n+1))
  for r in range(m):
    for c in range(n):
      result += img[r:r+M-m+1,c:c+N-n+1] * patn[r,c]
  return result 

def Laplace_Mask1(img, threshold):
  border_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE) 
  mask = np.array([[0,1,0],[1,-4,1],[0,1,0]])
  magnitude = conv(border_img, mask)
  return np.int8((magnitude > threshold))-np.int8((magnitude <= threshold*(-1)))

def Laplace_Mask2(img, threshold):
  border_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE) 
  mask = np.array([[1,1,1],[1,-8,1],[1,1,1]])
  magnitude = conv(border_img, mask)/3
  return np.int8((magnitude > threshold))-np.int8((magnitude <= threshold*(-1)))

def Minimum_variance_Laplacian(img, threshold):
  border_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE) 
  mask = np.array([[2,-1,2],[-1,-4,-1],[2,-1,2]])
  magnitude = conv(border_img, mask)/3
  return np.int8((magnitude > threshold))-np.int8((magnitude <= threshold*(-1)))

def Laplacian_of_Gaussian(img, threshold):
  border_img = cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_REPLICATE) 
  mask = LoG_mtx
  magnitude = conv(border_img, mask)
  return np.int8((magnitude > threshold))-np.int8((magnitude <= threshold*(-1)))

def Difference_of_Gaussian(img, threshold):
  border_img = cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_REPLICATE) 
  mask = DoG_mtx
  magnitude = conv(border_img, mask)
  return np.int8((magnitude > threshold))-np.int8((magnitude <= threshold*(-1)))

def zero_crossing(mtx):
  border_mtx = cv2.copyMakeBorder(mtx,1,1,1,1,cv2.BORDER_REPLICATE)
  M,N = mtx.shape
  cdt_part2 = np.zeros((M,N))
  for r in range(3):
    for c in range(3):
      cdt_part2 = np.minimum(border_mtx[r:r+M,c:c+N], cdt_part2)
  cdt = (mtx != 1)|(cdt_part2 != -1)
  return cdt*255

lena_gray = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

result = zero_crossing(Laplace_Mask1(lena_gray, 15))
cv2.imwrite("Laplace_Mask1_15.bmp", result)
result = zero_crossing(Laplace_Mask2(lena_gray, 15))
cv2.imwrite("Laplace_Mask2_15.bmp", result)
result = zero_crossing(Minimum_variance_Laplacian(lena_gray, 20))
cv2.imwrite("Minimum_variance_Laplacian_20.bmp", result)
result = zero_crossing(Laplacian_of_Gaussian(lena_gray, 3000))
cv2.imwrite("Laplacian_of_Gaussian_3000.bmp", result)
result = zero_crossing(Difference_of_Gaussian(lena_gray, 1))
cv2.imwrite("Difference_of_Gaussian_1.bmp", result)