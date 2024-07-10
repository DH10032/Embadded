import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from fastai.vision.all import *

import PIL

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

path = '/Users/dh/Downloads/8.png' # 사진 경로

img = cv2.imread(path, 1)

A_img = [ [(0.3*img[i][j][0]+0.59*img[i][j][1]+0.11*img[i][j][2]) for j in range(len(img[0]))] for i in range(len(img))]

New_img = A_img

# 명암 분리
for i in range(len(A_img)):
    for j in range(len(A_img[0])):
        New_img[i][j] = A_img[i][j]
        if A_img[i][j] > 200:
            New_img[i][j] = 130
        elif A_img[i][j] < 130:
            New_img[i][j] = 0


New_img = np.array(New_img)

kernel = [[1 for i in range(20)] for j in range(20)]
kernel = np.array(kernel)

A = 256 - New_img
blured = cv2.filter2D(A, -1, kernel, dst=None, anchor=None, delta=None, borderType=None)/(4*256)

F_list = list(blured)

F_arr = F_list

# 문자 분리 알고리즘
for i in range(len(F_list)):
    for j in range(len(F_list[0])):
        F_arr[i][j] = F_list[i][j]
        if F_list[i][j] < 35:
            print("x", F_arr[i][j] ,"\n")
            F_arr[i][j] = 0
        else:
            print("o", F_arr[i][j] ,"\n")
            F_arr[i][j] = 35

Fin = (np.array(F_arr))

df = pd.DataFrame(Fin) #Fin
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')


plt.imshow(df, cmap = 'gray')
plt.show()
