import torch.nn as nn
import torch.nn.functional as F


# 神经网络结构定义
class FLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer1 = nn.Conv1d(1, 10, 3)
        # self.conv_layer2 = nn.Conv1d(10, 5, 3)
        self.pool_layer1 = nn.MaxPool1d(5)
        self.fc1 = nn.Linear(150, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 14)

    def forward(self, x):
        # x=x.view(1,64,79)
        x = self.conv_layer1(x)
        # x=self.conv_layer2(x)
        x = self.pool_layer1(x)
        x = x.view(-1, 150)
        # x=self.flatten_layer(x)
        # x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)

        return output
