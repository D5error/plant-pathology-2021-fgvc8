# -*- coding: utf-8 -*-
from torchvision import transforms
import torch.jit
from torch import nn
import torch
import Plant_dataset
from models import *


if __name__ == "__main__":
    dataset_path = "./plant_dataset" # 数据集的路径
    checkpoint_path = "./save_model/checkpoint.pth" # 模型保存的路径
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小
        transforms.ToTensor(), # 将图像PIL转pytorch为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化（适用于预训练模型）
    ])
    model = LeNet5(num_classes=12) # 创建模型
    loss_function = nn.CrossEntropyLoss() # 定义损失函数（交叉熵损失）
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9) # 定义优化器(随机梯度下降法)
    train_or_test = "train" # 训练或者测试
    num_epoch = 100 # 训练轮数
    num_workers = 6 # 加载数据时的并行度
    batch_size = 32 # 批量大小

    # 运行
    Plant_dataset.start(dataset_path, transform, train_or_test, checkpoint_path, batch_size, num_workers, model, loss_function, optimizer, num_epoch)