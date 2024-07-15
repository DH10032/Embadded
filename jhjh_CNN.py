import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pathlib import Path
import random

# 경로 설정
path = Path('/home/jijijiho/다운로드/한국어글자체/syllables')  # 손글씨와 프린트 글씨가 있는 경로

# 하이퍼파라미터 설정
learning_rate = 0.00005  # 학습률 조정
num_epoch = 20  # 에포크
new_width = 200
new_height = 200
Num_Img = 1000
batch_size = 32

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, path, num_img):
        self.path = path
        self.num_img = num_img
        self.data = self.load_data()
    
    def load_data(self):
        HandWriting = sorted(list((self.path/'hand').glob('*')))
        Printed = sorted(list((self.path/'printed').glob('*')))
        
        transform = transforms.Compose([
            transforms.Resize((new_width, new_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 정규화 추가
        ])

        HandWriting_tensor = [transform(Image.open(file)) for file in random.sample(HandWriting, self.num_img)]
        Printed_tensor = [transform(Image.open(file)) for file in random.sample(Printed, self.num_img)]

        labels = [0] * len(HandWriting_tensor) + [1] * len(Printed_tensor)
        data = list(zip(labels, HandWriting_tensor + Printed_tensor))
        random.shuffle(data)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, image = self.data[idx]
        return image, label

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
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
            nn.Linear(10, 2),
        )
        
    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.size(0), -1)  # 배치 크기를 유지하며 평탄화
        out = self.fc_layer(out)
        return out

# 데이터 로드
dataset = CustomDataset(path, Num_Img)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 초기화
model = CNN().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
model.train()
for epoch in range(num_epoch):
    total_loss = 0
    model.train()
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epoch}], Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

# 모델 평가
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f'Accuracy of the model: {100 * correct / total:.2f}%')
