import os
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
EPOCHS = 1000
WAVEFORMS_INPUT = 7     # V1, V2, V3, dV1dn, dV2dn, dV3dn, I
OUTPUT_SIZE = 6         # G1, G2, G3, C1, C2, C3
BATCHSIZE = 150000
LAYER1_NODE = 64
LAYER2_NODE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
input("엔터를 누르면 계속합니다...")
dataPipe = dlr.MakeInputDatas(DATALEN, LEN_PER_ONECYCLE);

class MRM3PNet(nn.Module):
    def __init__(self):
        super(MRM3PNet, self).__init__()
        self.fc1 = nn.Conv1d(in_channels=WAVEFORMS_INPUT, out_channels=1, kernel_size=LEN_PER_ONECYCLE, padding=0, stride=1)
        self.fc2 = nn.Linear(in_features=DATALEN-LEN_PER_ONECYCLE+1, out_features=LAYER1_NODE)
        self.fc3 = nn.Linear(in_features=LAYER1_NODE, out_features=LAYER2_NODE)
        self.fc4 = nn.Linear(in_features=LAYER2_NODE, out_features=OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), -1)  # 텐서 차원을 맞추기 위해 필요
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
        
model = MRM3PNet().to(device)

# 모델이 저장된 경로에서 모델을 불러오기
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded existing model and continue training.")
else:
    print("No existing model found, starting training from scratch.")

# 손실 함수 및 최적화 알고리즘 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, data_loader, optimizer, criterion, epochs, model_path):
    index = 0
    while True:
        index += 1
        inputs, targets = data_loader.get_dataSets(BATCHSIZE)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if epoch == 0:
                startLoss = loss.item()
            loss.backward()
            optimizer.step()
            
            print(f'Index [ {index} ], Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.10f}')
            if loss.item() / startLoss < 0.000000001 or loss.item() < 0.0000001:
                break

try:
    train(model, dataPipe, optimizer, criterion, EPOCHS, model_path)
except KeyboardInterrupt:
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
