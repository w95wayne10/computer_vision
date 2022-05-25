import numpy as np
import matplotlib.pyplot as plt
import cv2
from mask import J, K, octogonal_kernel, N, LoG_mtx, DoG_mtx

def Dilation(img, pattern):
  bo = (pattern.shape[0]-1)//2
  border_img = cv2.copyMakeBorder(img, bo, bo, bo, bo, cv2.BORDER_REPLICATE)
  m,n = img.shape
  result = np.zeros(img.shape)
  for i in range(5):
    for j in range(5):
      if pattern[i,j] == 1:
        result = np.maximum(result,border_img[i:i+m,j:j+n])
  return result

def Erosion(img, pattern):
  bo = (pattern.shape[0]-1)//2
  border_img = cv2.copyMakeBorder(img, bo, bo, bo, bo, cv2.BORDER_REPLICATE)
  m,n = img.shape
  result = np.ones(img.shape)*255
  for i in range(5):
    for j in range(5):
      if pattern[i,j] == 1:
        result = np.minimum(result,border_img[i:i+m,j:j+n])
  return result

def Opening(img, pattern):
  return Dilation(Erosion(img, pattern), pattern)

def Closing(img, pattern):
  return Erosion(Dilation(img, pattern), pattern)

def gray_level_value_count(gray_img):
  value_count = [0]*256
  for row in gray_img:
    for pix in row:
      value_count[pix]+=1
  return value_count

def gray_level_value_count_chart(gray_img, out_fig_name):
  value_count = gray_level_value_count(gray_img)
  plt.figure(dpi=100)
  #plt.figure(dpi=500)
  plt.bar(list(range(256)),value_count,1)
  plt.title(out_fig_name[:-4])
  plt.xlabel("gray level value")
  plt.xlim(0,255)
  plt.ylabel("count")
  plt.show()
  #plt.savefig(out_fig_name)
  return

def gray_hist_equalization(gray_img):
  value_cumsum = []
  minlevel, maxlevel, sum_value = -1, -1, 0
  novalue = True
  value_count = gray_level_value_count(gray_img)
  for level, count in enumerate(value_count):
    sum_value += count
    value_cumsum.append(sum_value)
    if novalue and count>0:
      novalue=False
      minlevel = level
    if count>0:
      maxlevel = level

  new_value_count = [0]*256
  transfor_list = [0]*256
  Denominator = value_cumsum[maxlevel]-value_cumsum[minlevel]
  Numerator = lambda x: value_cumsum[x]-value_cumsum[minlevel]
  for level in range(minlevel, maxlevel+1):
    new_level = int(np.round(Numerator(level)*255/Denominator))
    transfor_list[level] = new_level
    new_value_count[new_level] += value_count[level]

  hist_equalization_lena_gray = gray_img.copy()
  for r,row in enumerate(hist_equalization_lena_gray):
    for c,_ in enumerate(row):
      hist_equalization_lena_gray[r,c] = transfor_list[hist_equalization_lena_gray[r,c]]
  return hist_equalization_lena_gray

def h(b,c,d,e):
  if b != c:
    return 's'
  else:
    if d==b and e==b:
      return 'r'
    else:
      return 'q'

def f(mtx):
  mtx_list = mtx.reshape(-1)
  qvalue = 0
  rvalue = 0
  for ads in [[4,5,2,1],[4,1,0,3],[4,3,6,7],[4,7,8,5]]:
    state = h(*mtx_list[ads])
    qvalue += 1 if state=='q' else 0
    rvalue += 1 if state=='r' else 0
  if rvalue==4:
    return 5
  else:
    return qvalue

def y(mtx):
  mtx_list = mtx.reshape(-1)
  if mtx_list[4] == 1:
    if 1 in mtx_list[[1,3,5,7]]:
      return 'p'
  return 'q'

def f2(mtx):
  mtx_list = mtx.reshape(-1)
  qvalue = 0
  rvalue = 0
  for ads in [[4,5,2,1],[4,1,0,3],[4,3,6,7],[4,7,8,5]]:
    state = h(*mtx_list[ads])
    qvalue += 1 if state=='q' else 0
  if qvalue==1:
    return 0
  else:
    return 255

def thinning_one_step(org_lena):
  check_table = np.zeros((66,66))
  check_table[1:-1,1:-1] = org_lena
  result = np.zeros((64,64))
  for r in range(64):
    for c in range(64):
      if check_table[r+1,c+1]>254:
        result[r,c] = int(np.around(f(check_table[r:r+3,c:c+3])))
  check_table2 = np.zeros((66,66))
  check_table2[1:-1,1:-1] = result
  result2 = np.zeros((64,64),str)
  for r in range(64):
    for c in range(64):
      if check_table2[r+1,c+1]>0.5:
        result2[r,c] = y(check_table2[r:r+3,c:c+3])
  new_lena = check_table[1:-1,1:-1]
  for r in range(64):
    for c in range(64):
      if result2[r,c]=='p':
        new_lena[r,c] = f2(check_table[r:r+3,c:c+3])
  return new_lena

def BoxFilter(img, size):
  bo = 1 if size == 3 else 2
  border_img = cv2.copyMakeBorder(img,bo,bo,bo,bo,cv2.BORDER_REPLICATE)
  m,n = lena_gray.shape
  total = np.zeros((m,n))
  for r in range(size):
    for c in range(size):
      total += border_img[r:r+m,c:c+n]
  result = total/(size**2)
  return np.around(result)

def MedianFilter(img, size):
  bo = 1 if size == 3 else 2
  border_img = cv2.copyMakeBorder(img,bo,bo,bo,bo,cv2.BORDER_REPLICATE)
  m,n = lena_gray.shape
  total = np.zeros((m,n,size**2))
  i = 0
  for r in range(size):
    for c in range(size):
      total[:,:,i] = border_img[r:r+m,c:c+n]
      i+=1
  total_sort = np.sort(total,2)
  result = total_sort[:,:,size**2//2]
  return result

def conv(img_org, patn):
  img = img_org.astype(float)
  m,n = patn.shape
  M,N = img.shape
  result = np.zeros((M-m+1,N-n+1))
  for r in range(m):
    for c in range(n):
      result += img[r:r+M-m+1,c:c+N-n+1] * patn[r,c]
  return result 

def spin(mtx):
  a,b,c = [0,0,1],[1,2,2],[0,1,2]
  return np.array([mtx[a,b],mtx[c,c],mtx[b,a]])

def Robert_Operator(img, threshold):
  m,n = img.shape
  magnitude = np.zeros((m,n))
  temp = img.astype(float)
  magnitude[0:m-1,0:n-1] = np.sqrt((temp[1:m,1:n]-temp[0:m-1,0:n-1])**2 + (temp[1:m,0:n-1]-temp[0:m-1,1:n])**2)
  return np.uint8((magnitude < threshold) * 255)

def Prewitt_Edge_Detector(img, threshold):
  border_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE) 
  p1 = np.array([[-1]*3,[0]*3,[1]*3])
  p2 = p1.T
  magnitude = np.sqrt(conv(border_img, p1)**2 + conv(border_img, p2)**2)
  return np.uint8((magnitude < threshold) * 255)

def Sobel_Operator(img, threshold):
  border_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE) 
  s1 = np.array([[-1,-2,-1],[0]*3,[1,2,1]])
  s2 = s1.T
  magnitude = np.sqrt(conv(border_img, s1)**2 + conv(border_img, s2)**2)
  return np.uint8((magnitude < threshold) * 255)

def FreiChen_Gradient_Operator(img, threshold):
  border_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
  s1 = np.array([[-1,-np.sqrt(2),-1],[0]*3,[1,np.sqrt(2),1]])
  s2 = s1.T
  magnitude = np.sqrt(conv(border_img, s1)**2 + conv(border_img, s2)**2)
  return np.uint8((magnitude < threshold) * 255)

def Kirsch_Compass_Operator(img, threshold):
  border_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE) 
  temp = np.array([[-3]*3]*3)
  temp[1,1] = 0
  temp[:,2] = 5
  magnitude = conv(border_img, temp)
  for i in range(7):
    temp = spin(temp)
    magnitude = np.maximum(conv(border_img, temp),magnitude)
  return np.uint8((magnitude < threshold) * 255)

def Robinson_Compass_Operator(img, threshold):
  border_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
  part = np.array([-1,0,1])
  temp = np.array([part,part*2,part])
  magnitude = conv(border_img, temp)
  for i in range(7):
    temp = spin(temp)
    magnitude = np.maximum(conv(border_img, temp),magnitude)
  return np.uint8((magnitude < threshold) * 255)

def Nevatia_Babu_5x5_operator(img, threshold):
  border_img = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE)
  
  magnitude = conv(border_img, N[0])
  for i in range(1,6):
    magnitude = np.maximum(conv(border_img, N[i]),magnitude)
  return np.uint8((magnitude < threshold) * 255)

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