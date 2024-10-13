import torch
import scipy.ndimage as nd


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

#将标签转换为独热编码（One Hot Encoding）。labels 是输入的标签，num_classes 是类别数量（默认为 10）。
#使用 torch.eye(num_classes) 创建一个单位矩阵，然后根据输入的标签返回对应的独热编码
def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

#将输入的图像 x 旋转指定的角度 deg。
#首先将输入展平为 28x28 的二维数组，然后使用 nd.rotate 进行旋转，reshape=False 表示输出的形状与输入相同。
#最后将旋转后的图像展平为一维数组并返回。
def rotate_img(x, deg):
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()
