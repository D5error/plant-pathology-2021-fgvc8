# -*- coding: utf-8 -*-
from torchvision import transforms
import torch.jit
from Plant_dataset import *
from models import *
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)


class BinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        # 损失
        loss = self.fn(logits, target)

        return loss

class FocalLoss(torch.nn.Module):
    '''
    alpha是平衡因子，用于调整正负样本的权重。
    gamma是调节因子，用于调整难易样本的权重。
    '''
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.small_num =  1e-12 # 防止在计算过程中出现数值不稳定性，特别是在取对数时出现无穷大或无效的情况

    def forward(self, logits, target):
        # Sigmoid激活后的概率
        probs = torch.sigmoid(logits)

        # 计算log(p_t)，其中p_t是模型对真实类别的预测概率
        # log(pt​) = target * log(probs) + (1−target) * log(1−probs)
        log_pt = target * torch.log(probs + self.small_num) + (1.0 - target) * torch.log(1.0 - probs + self.small_num)
        
        # 计算p_t
        pt = torch.exp(log_pt)

        # Focal Loss计算公式
        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt

        return torch.mean(focal_loss)


if __name__ == "__main__":
    # one-hot 标签
    labels_name = [
        "scab", # 癣病
        "healthy", # 健康
        "frog_eye_leaf_spot", # 蛙眼叶斑病
        "rust", # 锈病
        "complex", # 复合感染
        "powdery_mildew" # 白粉病
    ]

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小
        transforms.ToTensor(), # 将图像PIL转pytorch为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
    ])

    # 加载数据集
    train_dataset = Plant_dataset(
        "./plant_dataset/train/train_label.csv",
        "./plant_dataset/train/images",
        labels_name, transform)
    test_dataset = Plant_dataset(
        "./plant_dataset/test/test_label.csv",
        "./plant_dataset/test/images",
        labels_name, transform)
    validate_dataset = Plant_dataset(
        "./plant_dataset/val/val_label.csv",
        "./plant_dataset/val/images",
        labels_name, transform)

    # 模型
    LeNet5 = select_model("LeNet5")
    Vgg16 = select_model("Vgg16")
    Vgg19 = select_model("Vgg19")
    Deep_ViT = select_model("Deep_ViT")

    # 训练
    train_model = Deep_ViT # 使用的模型
    trainer = Trainer(
        train_dataset = train_dataset, # 训练集
        val_dataset = validate_dataset, # 验证集
        batch_size = 8, # 批量大小
        num_workers = 6, # 加载数据时的并行度
        model = train_model,
        optimizer = torch.optim.SGD(train_model.parameters(), lr=1e-5, momentum=0.8), # 优化器(随机梯度下降法)
        loss_function = torch.nn.BCEWithLogitsLoss(), # 损失函数
        save_path = r".\save_model", # 保存路径
    )
    trainer.train(
        num_epoch = 1000, # 训练轮数
        checkpoint_path = r".\save_model\Deep_ViT_epoch_36.pth", # 检查点
    )

    # 测试
    # test_model = Vgg16 # 使用的模型
    # tester = Tester(
    #     model = test_model,
    #     loss_function = BinaryCrossEntropyLoss(), # 损失函数
    # )
    # tester.test(
    #     checkpoint_path = "./save_model/Vgg16_epoch_35.pth", # 检查点
    #     batch_size = 128, # 批量大小
    #     num_workers = 6, # 加载数据时的并行度
    #     dataset = test_dataset, # 测试集
    # )

    # 图片预测
    # img_path = r"plant_dataset\train\images\ccb152972a7a6555.jpg" # 图片路径
    # model_path = "./save_model/LeNet5_epoch_364.pth" # 使用的模型路径
    # predict_img(img_path, transform, model, model_path, labels_name, optimizer)
