# -*- coding: utf-8 -*-
from tkinter import Image
from torchvision import transforms
import torch.jit
from torch import nn
import torch
from Plant_dataset import *
from models import *


if __name__ == "__main__":
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小
        transforms.ToTensor(), # 将图像PIL转pytorch为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
    ])

    # 参数
    dataset_path = "./plant_dataset" # 数据集的路径
    checkpoint_path = "./save_model/checkpoint.pth" # 模型保存的路径
    num_classes = 6 # 类别个数
    model = LeNet5(num_classes) # 创建模型
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9) # 定义优化器(随机梯度下降法)
    num_epoch = 1 # 训练轮数
    num_workers = 6 # 加载数据时的并行度
    batch_size = 16 # 批量大小
    threshold = 0.5 # 某标签的概率大于此阈值，则被分类为该标签
    labels_name = [  # 标签
        "scab", # 癣病
        "healthy", # 健康
        "frog_eye_leaf_spot", # 蛙眼叶斑病
        "rust", # 锈病
        "complex", # 复杂病（可能是多种病害的组合）
        "powdery_mildew" # 白粉病
    ]


##############################################
    # 开始训练或测试
    # train_or_test = "train"
    train_or_test = "test"
    start(dataset_path, transform, train_or_test, checkpoint_path, batch_size, num_workers, model, loss_function, optimizer, num_epoch, labels_name, threshold)

    # img_path = "./plant_dataset/val/images/8a3937a9ab9265a5.jpg"
    # model_path = "./save_model/checkpoint.pth"
    # predict_img(img_path, transform, model, num_classes, model_path, labels_name)
