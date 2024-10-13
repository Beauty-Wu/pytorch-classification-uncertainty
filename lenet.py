import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LeNet(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.use_dropout = dropout
        #定义了两个卷积层 conv1 和 conv2，分别将输入通道数从 1 转换为 20，再从 20 转换为 50，卷积核大小为 5。
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        #定义了两个全连接层 fc1 和 fc2，fc1 将输入的 20000 维特征映射到 500 维，fc2 将 500 维特征映射到 10 维（对应 10 个类别）
        self.fc1 = nn.Linear(20000, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        #输入 x 经过第一个卷积层 conv1，然后进行最大池化和 ReLU 激活。
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        #将特征图展平为一维向量。
        x = x.view(x.size()[0], -1)
        #过第一个全连接层 fc1，并应用 ReLU 激活。
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
