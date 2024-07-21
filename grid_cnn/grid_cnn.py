'''
[[원본 이미지 경로, [[grid 이미지 경로, x, y, status], [grid 이미지 경로, x, y, status]]], ... , [원본 이미지 경로, [[grid 이미지 경로, x, y, status], ]], ...]
'''

import torch
import torch.nn as nn
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# =======================================================================

path = '/home/ldh/Desktop'
grid = 30   # 단위 픽셀
output_folder_name = '/grid_img'

# =======================================================================
'''
일단 기존 모델 차용함
'''
class grid_model(nn.Module):
    def __init__(self):
        super(grid_model, self).__init__()

        self.convolution_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 12 * 12, 1000),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.convolution_layer(x)
        out = out.view(out.size(0), -1)  # 배치 크기를 유지하며 평탄화
        out = self.fc_layer(out)
        return out
    
# =======================================================================

if torch.cuda.is_available(): #is_available : CUDA가 현재 사용 가능한지를 bool 결과로 반환
    DEVICE = torch.device("cuda")
    print(DEVICE, torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print(DEVICE)

# =======================================================================

learning_rate = 0.00005  # 학습률 조정
num_epoch = 20  # 에포크
new_width = 200
new_height = 200
Num_Img = 1000
batch_size = 5

# =======================================================================   데이터 준비하기
with open('data.json', 'r') as f:
    json_data = json.load(f)

transform = transforms.Compose([    # 이미지 변환을 압축시킨 함수
            transforms.Resize((new_width, new_height)),
            transforms.ToTensor(),
            # transforms.Normalize(mean = [0, 0, 0], std = [1, 1, 1])  # 정규화 추가 0~1 사이로 맞춤
        ])

dataset = list((data_path[3], transform(Image.open(Path(data_path[0])))) for data_path in json_data)

train_size = int(0.3 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("train_dataset : ", len(train_dataset))
print("val_dataset : ", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# =======================================================================


# 위에 클래스에서 정의한 모델을 가져옴
# 모델 초기화
model = grid_model()
model = model.to(DEVICE)                # 모델을 Device로 이동
model.train()

loss_func = nn.CrossEntropyLoss()       # 손실함수를 CrossEntropyLoss를 사용함 (좋은지는 모르겠음)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # backword를 하면 파라미터 값을 조정하는 함수

for i in range(num_epoch):
    for y, x in train_loader:

        x = x.to(DEVICE)      # tensor를 x에 저장
        y = y.to(DEVICE)

        optimizer.zero_grad()
        
        output = model.forward(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
    
    print("{0:02d} 번째  loss :  {1:0.2f}".format(i, loss.item()))

correct = 0
total = 0

# evaluate model
model.eval()

fail_img = []

with torch.no_grad():
    for label, img in val_loader:
        image = img.to(DEVICE)  # tensor를 image에 저장
        label = label.to(DEVICE)

        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)  # , 생략의 의미 제거
        total += label.size(0)
        correct += (predicted == label).cpu().sum().item()

        for i in range(len(label)):  # y가 아닌 label 사용
            if label[i] != predicted[i]:
                fail_img.append((label[i], image[i]))

    accuracy = 100 * float(correct) / total
    print(f'Accuracy of the model: {accuracy:.2f}%')
