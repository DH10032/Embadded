'''
평탄화 작업을 위해 데이터 셋 다시 구성해야 함
'''


import torch                                        # 기본 import(텐서와 텐서 계산 라이브러리)(그냥 뭔가 불안해서 부름)
from torch.utils.data import Dataset, DataLoader    # DataLoader 함수를 위한 import
import matplotlib.pyplot as plt                     # show()를 위한 import
import numpy as np                                  # 행렬 라이브러리 아직 크게 필요없음 (지워도 됨)
import os                                           # 운영 체제와 상호작용하는 라이브러리. 바로 아래 코드를 위해 부름 (maybe?)서
from PIL import Image                               # Image.open()을 위해 부름
import torch.nn as nn                               # 모든 신경망들의 Base class이다. (굉장히 중요) (손실함수, 컨볼루션 층, 풀링 층, 활성화 함수 등을 위 사용)
import torch.optim as optim                         # 최적화 알고리즘들이 포함됨(gradient를 통해 정답에 가까이 가게 됨. 이용하는 기법을 선택할 수 있는데, SGD, Moment, adams 등이 있음: https://amber-chaeeunk.tistory.com/23)
import torch.nn.init as init                        # 텐서에 초기값을 줌
import torchvision.datasets as datasets             # 이미지 데이터셋 집합체
import torchvision.transforms as transforms         # 이미지 변환 툴
from torch.utils.data import DataLoader             # 학습 및 배치로 모델에 넣어주기 위한 툴
import matplotlib.pyplot as plt                     # 이미지 출력을 위함

from pathlib import Path
import random

'''
순전파 : 입력값 X부터 순차적으로 레이어가 진행
역전파 : 출력층부터 역으로 레이어가 진행

1.
역전파 과정에서 SGD를 사용한다. 이 때 acitvation function을 sigmoid 함수를 사용할 경우,  Vanishing Gradient Problem이 발생한다.
따라서 Relu 함수를 통해 이 문제를 극복하며, SGD의 속도를 올릴 수 있다.

결론 : activation f를 Relu쓸거임

2. DataLoad (일단 적어는 두는데 쓸 곳은 없음)
mnist_train = datasets.MNIST(root="../Data/", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)

    root="원하는 경로"
    - root는 우리가 데이터를 어디에다가 저장하고, 경로로 사용할지를 정의해줍니다.

    train=True(또는 False)
    - train은 우리가 지금 정의하는 데이터가 학습용인지 테스트용인지 정의해줍니다.

    transform=transforms.ToTensor()
    - 데이터에 어떠한 변형을 줄 것인가를 정의해줍니다.음
        해당 transforms.ToTensor()의 경우, 모델에 넣어주기 위해 텐서 변환을 해줌을 의미합니다.

    target_transform=None
    - 라벨(클래스)에 어떠한 변형을 줄 것인가를 정의해줍니다.

    download=True
    - 앞에서 지정해준 경로에 해당 데이터가 없을 시 다운로드하도록 정의해줍니다.


    
idea
프린트된 글자 객체와 아닌 객체 구분

1. 배경을 최대한 정리한다.
2. 후보 바운더리를 만든다.
3. 후보들을 분리한다.
    프린트된 글자 객체, 낙서 객체, 배경 객체


목표 : 손글씨와 프린트 글씨를 구분하게 만들기

'''
# ===========================================================================

path = Path('/home/ldh/Desktop')  # (주의!!) 컴퓨터에서 손글씨 폴더와 프린트 글씨가 있는 경로

# Num_Img = 100            # 일괄적으로 처리하는 데이터 양 (이거는 지금 당장 사용 못함)
learning_rate = 0.002      # backword 과정에서 SGD를 할 때 중요하게 쓰임
num_epoch = 10             # 전체 데이터셋을 학습 하는 횟수
new_width = 200            # 이미지 사이즈 통일 아래 cnn 계산할 때를 위함
new_height = 200
Num_Img = 3000             # 랜덤으로 가져오는 이미지 개수

# ===========================================================================
# 만약 현재 장치가 GPU를 지원하지 못한다면, CPU를 Device로 지정
# 맨위 출력을 통해 확인 가능 (그냥 무시해도 되는 부분임)
if torch.cuda.is_available(): #is_available : CUDA가 현재 사용 가능한지를 bool 결과로 반환
    DEVICE = torch.device("cuda")
    print(DEVICE, torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print(DEVICE)

# ===========================================================================
class DL(Dataset):
    def __init__(self, df):
        self.df = df                                           # 이게 이해가 안가면 class 공부 ㄱㄱ
    
    def T(self):
        HandWriting = sorted(list((path/'hand').glob('*')))     # ls가 안됨 원인은 정확히 알 수 없음
        Printed = sorted(list((path/'printed').glob('*')))

        # 폴더의 이미지 로드
        transform = transforms.Compose([    # 이미지 변환을 압축시킨 함수
            transforms.ToTensor(),  # 이미지를 Torch 텐서로 변환
        ])

        '''
        각각 Num_Img개 만큼 랜덤으로 가져옴
        이게 싫고 전부 가져오고 싶다면 아래 코드를 쓰면됨 시간이 아주아주 오래 걸릴거임
        본인은 짜증나서 랜덤으로 돌림
        나중에 필요하면 아래 코드로 바꿀 예정

        HandWriting_tensor = [transform(Image.open(file)).to(DEVICE) for file in HandWriting]
        Printed_tensor = [transform(Image.open(file)).to(DEVICE) for file in Printed]
        '''

        # 학습 과정에서는 for 문을 남발해도 큰 지장은 없습니다.
        HandWriting_tensor = [transform(Image.open(file).resize((new_width, new_height))).to(DEVICE) for file in random.sample(HandWriting, Num_Img)]
        Printed_tensor = [transform(Image.open(file).resize((new_width, new_height))).to(DEVICE) for file in random.sample(Printed, Num_Img)]

        # 두 tensor 합치고 label 붙이기
        label = [torch.tensor([0.0, 1.0]) for i in range(len(HandWriting_tensor))] + [torch.tensor([1.0, 0.0]) for i in range(len(Printed_tensor))]
        result = list(zip(label, HandWriting_tensor + Printed_tensor))

        return result
    
    # 이미지 정규화 함수 만들기 (수렴속도와 안정성) 0~1

# ===========================================================================
'''
기본적으로 신경망을 정의할 때, nn.Module을 상속하는 경우와 아닌 경우 두개가 존재한다.
nn.Module을 상속받지 않는 경우, 매운 단순한 모델을 만들때 사용한다.

nn.Module을 상속받는 경우, 기본적으로 __init__과 forward가 포함되어야 한다. override(재정의)를 할 필요가 있음 (걍 외우셈)
__inti__ 모델에서 사용하는 모듈(nn.linear, nn.Conv2d), 활성화 함수(sigmoid, Relu)등을 정의한다.
forward는 모델에서 실행되어야 하는 연산을 정의한다.

- 참고 출처 : https://resultofeffort.tistory.com/81
'''

# 기본적인 구조
class CNN(nn.Module):       # nn.module class 상속
    def __init__(self):
    	# super함수는 CNN class의 부모 class인 nn.Module을 초기화
        super(CNN, self).__init__()                                     # 계층(layer) 초기화
        
        # convolution layer
        self.layer = nn.Sequential(                                     # 각 모듈을 순차적으로 실행한다.
            
            # [3, 150, 150]
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3),     # convolution 연산 (정말 놀랍게도 이미지 Tensor의 shape를 보니 RGB 이미지네요. 따라서 in_channels이 3입니다.)(상세 설명: https://blog.joonas.io/196)
            nn.ReLU(),                                                  # convoultion 연산과 활성화 함수 처리 사이에 padding이 필요하다면 nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1)
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2,stride=2),                       # 원래 vgg16은 3x3 커널이지만, 3x3으로 하기에 이미지가 너무 작아서 2x2로 진행->패딩을 사용하면 3x3 가능하려나?

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2,stride=2)

        )

        # Fully Connected Layer
        self.fc_layer = nn.Sequential(
        	# [1, 64*33*33] -> [1, 16]
            nn.Linear(256*3*3, 1000),
            nn.ReLU(),
            # [1,16] -> [1,2]
            nn.Linear(1000, 100),
            nn.ReLU(),

            nn.Linear(100, 10),
            nn.ReLU(),

            nn.Linear(10, 2)
            
        )
        
    def forward(self,x):
    	# self.layer에 정의한 연산 수행
        out = self.layer(x)
        out = out.unsqueeze(0)
        out = out.view(1, -1)     # torch.tensor.view(): 메모리 복사 없이, 같은 텐서를 다른 모양으로 보여줌(예를 들어, stride를 다르게 한다던가)). -1로 하면 자동 지정. 즉, 1행만 보여 줌.
        # self.fc_layer 정의한 연산 수행    
        out = self.fc_layer(out)
        return out

# ================================Main code==================================


Img_Train_List = DL(path).T()           # 정리하면 Img_Train_List[0]에는 (0, Tensor[...])가 있습니다.
Img_Test_List = DL(path).T()

# 위에 클래스에서 정의한 모델을 가져옴
# 모델 초기화
model = CNN()
model = model.to(DEVICE)                # 모델을 Device로 이동. torch.nn.Module.to() (https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to) DEVICE는 일반적으로 cuda 또는 cpu임.

print(model)

loss_func = nn.CrossEntropyLoss()       # 손실함수를 CrossEntropyLoss를 사용함 (좋은지는 모르겠음)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # backword를 하면 파라미터 값을 조정하는 함수

for i in range(num_epoch):
    for data in Img_Train_List:
        x = data[1].to(DEVICE)      # tensor를 x에 저장. torch.tensor.to() (https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to) DEVICE는 일반적으로 cuda 또는 cpu임.
        y= data[0].to(DEVICE)

        optimizer.zero_grad()       # grad 값을 0으로 초기화 후 역전파 -> grad값이 다음 epopc에 영향 미치는 것 방지
        
        output = model.forward(x)   # 하위 클래스의 메소드인 forward. 특징 추출(컨볼루션(+패딩), 활성화 함수 처리, 풀링)
        print(output)
        loss = loss_func(output[0],y)  # nn.CrossEntropyLoss에 인자 대입. 
        loss.backward()
        optimizer.step()

correct = 0
total = 0

# evaluate model
model.eval()

with torch.no_grad():
    total = 0
    for data in Img_Test_List:
        New_x = data[1].to(DEVICE)      # tensor를 x에 저장. torch.tensor.to() (https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to)
        New_y= data[0].to(DEVICE)       

        output = model.forward(New_x)   # 하위 클래스의 메소드인 forward. 특징 추출(컨볼루션(+패딩), 활성화 함수 처리, 풀링)

        # torch.max함수는 (최댓값,index)를 반환 -> dim = 1이면 행(가로) 중 최대, dim = 2이면 열(세로) 중 최대, dim = 3이면 높이 중 최대값을 모아 텐서로 반환.(+ 각 최대값의 인덱스)
        output_index = torch.max(output, dim=1)[1]    
        ''' ex) tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
                [ 1.1949, -1.1127, -2.2379, -0.6702],
                [ 1.5717, -0.9207,  0.1297, -1.8768],
                [-0.6172,  1.0036, -0.6060, -0.2432]]) -> torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1])) '''                                                      

        # 전체 개수 += 라벨의 개수
        total += 1.0
        
        # 도출한 모델의 index와 라벨이 일치하면 correct에 개수 추가
        if New_y[output_index[0].item()].item():
            correct += 1
        
    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))
