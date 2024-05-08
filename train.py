import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dataLoader as dlr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

model_path = 'mnist_model.pth'

LEN_PER_ONECYCLE = 100
DATALEN = 300
LEARNING_RATE = 0.001
EPOCHS = 100
WAVEFORMS_INPUT = 7     # V1, V2, V3, dV1dn, dV2dn, dV3dn, I
OUTPUT_SIZE = 6         # G1, G2, G3, C1, C2, C3
BATCHSIZE = 100000
LAYER1_NODE = LEN_PER_ONECYCLE
LAYER2_NODE = LEN_PER_ONECYCLE
isOperating = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("torch version: ", torch.__version__)
print("cuda version: ", torch.version.cuda)
print("cudnn version: ", torch.backends.cudnn.version())
dataPipe = dlr.MakeInputDatas(DATALEN, LEN_PER_ONECYCLE);

class MRM3PNet(nn.Module):
    def __init__(self):
        super(MRM3PNet, self).__init__()
        self.fc1 = nn.Conv1d(in_channels=WAVEFORMS_INPUT, out_channels=1, kernel_size=LEN_PER_ONECYCLE, padding=0, stride=1)
        self.fc2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=DATALEN-LEN_PER_ONECYCLE+1, out_features=LAYER1_NODE)
        self.fc4 = nn.Tanh()
        self.fc5 = nn.Linear(in_features=LAYER1_NODE, out_features=LAYER2_NODE)
        self.fc6 = nn.Tanh()
        self.fc7 = nn.Linear(in_features=LAYER2_NODE, out_features=OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), -1)  # 텐서 차원을 맞추기 위해 필요
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return x
        
model = MRM3PNet().to(device)

# 모델이 저장된 경로에서 모델을 불러오기
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded existing model and continue training.")
else:
    print("No existing model found, starting training from scratch.")

# 손실 함수 및 최적화 알고리즘 설정
criterion = nn.MSELoss() #nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE) #optim.Adam(model.parameters(), lr=LEARNING_RATE)

x = torch.tensor([1.0, 2.0, 3.0]).cuda()
print("cuda test: ", x)
input("엔터를 누르면 시작합니다...")

def train(model, data_loader, optimizer, criterion, epochs):
    index = 0
    while True:
        index += 1
        inputs, targets = data_loader.get_dataSets(BATCHSIZE)
        inputs = torch.from_numpy(np.array(inputs)).float().to(device)
        targets = torch.from_numpy(np.array(targets)).float().to(device)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if epoch == 0:
                startLoss = loss.item()

            print(f'Index [ {index} ], Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.10f}')
            if loss.item() / startLoss < 0.000000001 or loss.item() < 0.0000001:
                break
            isOperating=1
            loss.backward()
            optimizer.step()
            isOperating=0
            


try:
    train(model, dataPipe, optimizer, criterion, EPOCHS)
except KeyboardInterrupt:
    while(isOperating==1):
        time.sleep(1)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
