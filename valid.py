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
BATCHSIZE = 100000
LAYER1_NODE = LEN_PER_ONECYCLE
LAYER2_NODE = LEN_PER_ONECYCLE

dataPipe = dlr.MakeInputDatas(DATALEN, LEN_PER_ONECYCLE);
dataPipe.regen();
data = dataPipe.get_data();
target = dataPipe.get_target();

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
    inputs, targets = data_loader.get_dataSets(BATCHSIZE)
    inputs = torch.from_numpy(np.array(inputs)).float().to(device)
    targets = torch.from_numpy(np.array(targets)).float().to(device)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    print(f'True values: {targets}')
    print(f'Output values: {outputs}')
    print(f'loss: {loss}')

valid(model, dataPipe)
