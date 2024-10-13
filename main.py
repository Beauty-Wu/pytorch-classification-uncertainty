import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import argparse
from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding
from data import dataloaders, digit_one
from train import train_model
from test import rotating_image_classification, test_single_image
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from lenet import LeNet


def main():
    #使用 argparse 库解析用户输入的命令行参数
    parser = argparse.ArgumentParser()#创建解析器：初始化一个 ArgumentParser 对象，用于存储和管理命令行参数。
    #创建一个互斥组，确保在命令行中只能选择一个选项（例如，训练、测试或示例）。required=True 表示至少必须选择一个选项。
    mode_group = parser.add_mutually_exclusive_group(required=True)
    #添加参数：train：用于训练模型的标志。test：用于测试模型的标志。examples：用于展示 MNIST 数据的标志。
    #action="store_true"：表示如果该参数被指定，则将其值设置为 True，否则为 False。
    mode_group.add_argument("--train", action="store_true", help="To train the network.")
    mode_group.add_argument("--test", action="store_true", help="To test the network.")
    mode_group.add_argument("--examples", action="store_true", help="To example MNIST data.")
    parser.add_argument(
        "--epochs", default=10, type=int, help="Desired number of epochs."
    )
    parser.add_argument(
        "--dropout", action="store_true", help="Whether to use dropout or not."
    )
    parser.add_argument(
        "--uncertainty", action="store_true", help="Use uncertainty or not."
    )
    #添加不确定性类型参数：mse：选择均方误差作为损失函数。digamma：选择 digamma 损失函数。log：选择对数损失函数。
    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument(
        "--mse",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.",
    )
    uncertainty_type_group.add_argument(
        "--digamma",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.",
    )
    uncertainty_type_group.add_argument(
        "--log",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.",
    )
    #调用 parse_args() 方法解析命令行输入的参数，并将结果存储在 args 对象中。用户可以通过 args 访问各个参数的值。
    args = parser.parse_args()
    #如果用户选择 --examples 参数，将从验证数据集中随机选择几张图像，并将其显示为示例图像，最后保存为文件。
    if args.examples:
        examples = enumerate(dataloaders["val"])
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig("./images/examples.jpg")
#如果用户选择 --train，则初始化模型并根据参数设置选择要使用的损失函数（包括不确定性损失）。
    elif args.train:
        num_epochs = args.epochs
        use_uncertainty = args.uncertainty
        num_classes = 10

        model = LeNet(dropout=args.dropout)
        #根据用户输入的不确定性参数选择损失函数
        if use_uncertainty:#如果启用了不确定性，则从相应的损失函数中选择；
            if args.digamma:
                criterion = edl_digamma_loss
            elif args.log:
                criterion = edl_log_loss
            elif args.mse:
                criterion = edl_mse_loss
            else:
                parser.error("--uncertainty requires --mse, --log or --digamma.")
        else:#否则使用交叉熵损失函数
            criterion = nn.CrossEntropyLoss()
        #然后初始化 Adam 优化器。
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        device = get_device()
        model = model.to(device)
        #调用 train_model 函数来训练模型
        model, metrics = train_model(
            model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            scheduler=exp_lr_scheduler,
            num_epochs=num_epochs,
            device=device,
            uncertainty=use_uncertainty,
        )
        #将训练后的模型状态和优化器状态保存为字典，方便后续使用或恢复训练。
        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        #根据不同的损失函数保存模型的状态，例如使用 digamma、log 或 mse 时分别保存不同的文件。
        if use_uncertainty:
            if args.digamma:
                torch.save(state, "./results/model_uncertainty_digamma.pt")
                print("Saved: ./results/model_uncertainty_digamma.pt")
            if args.log:
                torch.save(state, "./results/model_uncertainty_log.pt")
                print("Saved: ./results/model_uncertainty_log.pt")
            if args.mse:
                torch.save(state, "./results/model_uncertainty_mse.pt")
                print("Saved: ./results/model_uncertainty_mse.pt")

        else:
            torch.save(state, "./results/model.pt")
            print("Saved: ./results/model.pt")
    #如果用户选择 --test，将加载保存的模型状态并评估模型。根据用户选择的不确定性设置加载相应的模型。
    elif args.test:

        use_uncertainty = args.uncertainty
        device = get_device()
        model = LeNet()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())

        if use_uncertainty:
            if args.digamma:
                checkpoint = torch.load("./results/model_uncertainty_digamma.pt")
                filename = "./results/rotate_uncertainty_digamma.jpg"
            if args.log:
                checkpoint = torch.load("./results/model_uncertainty_log.pt")
                filename = "./results/rotate_uncertainty_log.jpg"
            if args.mse:
                checkpoint = torch.load("./results/model_uncertainty_mse.pt")
                filename = "./results/rotate_uncertainty_mse.jpg"

        else:
            checkpoint = torch.load("./results/model.pt")
            filename = "./results/rotate.jpg"

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model.eval()#将模型设置为评估模式，这样在推理时会禁用 dropout 和 batch normalization 等训练时特有的操作。

        rotating_image_classification(
            model, digit_one, filename, uncertainty=use_uncertainty
        )
        #别对两张指定的图像（one.jpg 和 yoda.jpg）进行测试，使用加载的模型进行推理，并根据需要处理不确定性。
        test_single_image(model, "./data/one.jpg", uncertainty=use_uncertainty)
        test_single_image(model, "./data/yoda.jpg", uncertainty=use_uncertainty)


if __name__ == "__main__":
    main()
