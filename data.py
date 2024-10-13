import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#使用 MNIST 类加载训练数据集，指定数据存储路径
data_train = MNIST("./data/mnist",
                   download=True,
                   train=True,
                   transform=transforms.Compose([transforms.ToTensor()]))#将图像转换为张量格式，以便于后续处理。
#加载验证数据：
data_val = MNIST("./data/mnist",
                 train=False,
                 download=True,
                 transform=transforms.Compose([transforms.ToTensor()]))
#使用 DataLoader 创建训练数据的加载器，设置批量大小为 1000，
#shuffle=True 表示在每个 epoch 开始时打乱数据，num_workers=8 表示使用 8 个子进程加载数据。
dataloader_train = DataLoader(
    data_train, batch_size=1000, shuffle=True, num_workers=8)
dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=8)
#dataloaders 字典：将训练和验证数据加载器存储在一个字典中，方便后续访问。
dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}

digit_one, _ = data_val[5]
