import torch
import torch.nn as nn
import dataLoader as dlr
dataPipe = dlr.MakeInputDatas;

data = dataPipe(400,100).get_data();

print(data);