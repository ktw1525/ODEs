import os
import torch
import torch.nn as nn
import torch.optim as optim
import dataLoader as dlr
import matplotlib.pyplot as plt

model_path = 'mnist_model.pth'

LEN_PER_ONECYCLE = 100
DATALEN = 300
LEARNING_RATE = 0.001
EPOCHS = 1000
WAVEFORMS_INPUT = 7     # V1, V2, V3, dV1dn, dV2dn, dV3dn, I
INPUT_SIZE = DATALEN * WAVEFORMS_INPUT
WAVEFORMS_VECTORS = WAVEFORMS_INPUT*2
MATELEMENTS_INPUT = (WAVEFORMS_INPUT**2)*2
MATELEMENTS = (WAVEFORMS_INPUT**2)
BATCHSIZE = 1000
OUTPUT_SIZE = 6         # G1, G2, G3, C1, C2, C3

dataPipe = dlr.MakeInputDatas(DATALEN, LEN_PER_ONECYCLE);
dataPipe.regen();
data = dataPipe.get_data();
target = dataPipe.get_target();

class MultiplyLayer(nn.Module):
    def forward(self, x):
        X1 = x.repeat(len(x),1)
        X2 = X1.transpose(0,1)
        return torch.matmul(X2, X1).reshape(-1,)

class MRM3PNet(nn.Module):
    def __init__(self):
        super(MRM3PNet, self).__init__()
        self.ly1 = nn.Linear(INPUT_SIZE, INPUT_SIZE*2)
        self.ly2 = nn.Conv1d(1, WAVEFORMS_VECTORS, kernel_size=LEN_PER_ONECYCLE, stride=1, padding=0)
        self.ly3 = nn.Linear(INPUT_SIZE*2-LEN_PER_ONECYCLE+1, WAVEFORMS_VECTORS) # WAVEFORMS_VECTORS, MATELEMENTS_INPUT
        self.ly4 = MultiplyLayer()
        self.ly5 = nn.Linear(WAVEFORMS_VECTORS*WAVEFORMS_VECTORS, OUTPUT_SIZE)

    def forward(self, input):
        ly1=self.ly1(input)
        ly2=self.ly2(ly1)
        ly3=self.ly3(ly2)
        ly4=self.ly4(ly3)
        out=self.ly5(ly4)
        return out
        
model = MRM3PNet()

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
    index=0
    while(1):
        data_loader.regen()
        index=index+1
        for epoch in range(epochs):
            inputs = data_loader.get_data()
            targets = data_loader.get_target()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Index [ {index} ], Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            if (loss.item()<0.001):
                break

try:
    train(model, dataPipe, optimizer, criterion, EPOCHS, model_path)
except KeyboardInterrupt:
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
