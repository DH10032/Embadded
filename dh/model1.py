import torch                                        # 기본 import (그냥 부름)
from torch.utils.data import Dataset, DataLoader    # DataLoader 함수를 위한 import
import matplotlib.pyplot as plt                     # show()를 위한 import
import numpy as np                                  # 행렬 라이브러리
import os                                           # 잘은 모름 바로 아래 코드를 위해 부름 (maybe?)

# ===========================================================================

if torch.cuda.is_available(): #is_available : CUDA가 현재 사용 가능한지를 bool 결과로 반환
    DEVICE = torch.device("cuda")
    print(DEVICE, torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print(DEVICE)   
# ===========================================================================

# 이미지 폴더 설정
path = '/Users/dh/E-prj/cnn/Img/'

# 가중치 초기 설정 (랜덤)
size = (3, 3)
w = torch.rand(size)        # 처음 행렬은 랜덤으로 설정

# 


