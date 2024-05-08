import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import dataLoader as dlr
import matplotlib.pyplot as plt

model_path = 'mnist_model.pth'

LEN_PER_ONECYCLE = 100
DATALEN = 300
LEARNING_RATE = 0.001
EPOCHS = 100
WAVEFORMS_INPUT = 7     # V1, V2, V3, dV1dn, dV2dn, dV3dn, I
OUTPUT_SIZE = 6         # G1, G2, G3, C1, C2, C3
BATCH_SIZE = 1
KERNEL_SIZE = 3
LAYER1_NODE = 64
LAYER2_NODE = 64
LAYER3_NODE = 64

THREADNUM = 4
torch.set_num_threads(THREADNUM)
print('Use ', THREADNUM, ' threads')

dataPipe = dlr.MakeInputDatas(DATALEN, LEN_PER_ONECYCLE);
dataPipe.regen();
data = dataPipe.get_data();
target = dataPipe.get_target();

class MRM3PNet(nn.Module):
    def __init__(self):
        super(MRM3PNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=WAVEFORMS_INPUT, out_channels=LAYER1_NODE, kernel_size=KERNEL_SIZE, padding=1)
        self.conv2 = nn.Conv1d(in_channels=LAYER1_NODE, out_channels=LAYER2_NODE, kernel_size=KERNEL_SIZE, padding=1)
        self.lstm = nn.LSTM(input_size=DATALEN, hidden_size=LAYER3_NODE, num_layers=2, batch_first=True)
        self.fc = nn.Linear(LAYER3_NODE, OUTPUT_SIZE)  # 6 출력값

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 시퀀스의 마지막 아웃풋만 사용
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = MRM3PNet().to(device)

# 모델이 저장된 경로에서 모델을 불러오기
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded existing model and continue training.")
else:
    print("No existing model found, starting training from scratch.")

# 손실 함수 및 최적화 알고리즘 설정
criterion = nn.MSELoss()

def valid(model, data_loader):
    data_loader.regen()
    inputs, targets = data_loader.get_dataSets(BATCH_SIZE)
    inputs = torch.from_numpy(np.array(inputs)).float().to(device)
    targets = torch.from_numpy(np.array(targets)).float().to(device)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    print(f'True values: {targets}')
    print(f'Output values: {outputs}')
    print(f'loss: {loss}')

valid(model, dataPipe)
