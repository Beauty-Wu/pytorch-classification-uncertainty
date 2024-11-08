import torch
import torch.nn as nn
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence


def train_model(
    model,
    dataloaders,
    num_classes,
    criterion,#损失函数
    optimizer,
    scheduler=None,#可选参数
    num_epochs=25,
    device=None,
    uncertainty=False,
):

    since = time.time()

    if not device:
        device = get_device()
#保存当前模型的状态字典（即模型的参数和权重），并初始化最佳准确率
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
#初始化三个字典，分别用于存储训练过程中的损失、准确率和证据信息
    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}
    evidences = {"evidence": [], "type": [], "epoch": []}
#可视化
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 模型的训练和验证阶段切换模型状态
        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0#累积当前训练批次中的损失值，以便后续计算平均损失。
            running_corrects = 0.0#累积当前训练批次中正确预测的样本数量，以便后续计算准确率。
            correct = 0#记录模型在当前批次或整个训练过程中正确分类的样本数

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # 每个训练步骤开始前需要清空之前的梯度。
                optimizer.zero_grad()

                # forward
                # 上下文管理器用于控制是否进行梯度计算。只有在训练阶段（phase 为 "train"）时，才会计算梯度。验证阶段（phase 为 "val"）不需要计算梯度
                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:#变量为真，则进入代码块
                        y = one_hot_embedding(labels, num_classes)#调用 one_hot_embedding 函数，将 labels（标签）转换为独热编码（one-hot encoding），并且指定类别数量 num_classes
                        y = y.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)#从模型输出中获取预测结果 preds。这一步会找出每个样本的最大输出值对应的索引，即预测的类别。
                        loss = criterion(#计算损失
                            outputs, y.float(), epoch, num_classes, 10, device
                        )
#比较预测值和真实标签来计算模型的准确率，同时还计算了一个与证据相关的值
                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))#比较预测值 preds 和真实标签 labels，返回一个布尔张量后进行处理
                        acc = torch.mean(match)
                        #SL中变量获取
                        evidence = relu_evidence(outputs)#证据e
                        alpha = evidence + 1#dirichlet参数阿尔法
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)#不确定性u

                        total_evidence = torch.sum(evidence, 1, keepdim=True)#证据总和S
                        mean_evidence = torch.mean(total_evidence)
                        #对给定的证据数据进行分析，以获取在成功匹配和失败匹配情况下的平均证据值。
                        mean_evidence_succ = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match
                        ) / torch.sum(match + 1e-20)
                        mean_evidence_fail = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * (1 - match)
                        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics在训练模型时，持续跟踪总损失和总正确预测样本数量。
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if scheduler is not None:#检查调度器是否存在。如果调度器是 None，则意味着没有定义调度器，后面的代码将不会执行
                if phase == "train":
                    scheduler.step()#调用调度器的 step() 方法。这个方法通常用于更新学习率或其他训练参数，以便在训练过程中进行调整。

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            losses["loss"].append(epoch_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc.item())
            accuracy["epoch"].append(epoch)
            accuracy["phase"].append(phase)

            print(
                "{} loss: {:.4f} acc: {:.4f}".format(
                    phase.capitalize(), epoch_loss, epoch_acc
                )
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)

    return model, metrics
