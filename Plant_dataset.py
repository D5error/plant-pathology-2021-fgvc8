# -*- coding: utf-8 -*-
import os
import time
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

# 加载数据集
class Plant_dataset:
    def __init__(self, csv_path, imgs_path, labels_name, transform):
        self.__transform = transform
        self.path = imgs_path
        self.__imgs_path = []

        csv = pd.read_csv(csv_path)
        csv = self.one_hot_encode(csv, labels_name)

        # 提取图像路径和标签
        self.__imgs_path, self.__labels = [], []
        for _, row in csv.iterrows():
            image_path = os.path.join(imgs_path, row['images'])
            self.__imgs_path.append(image_path)

            # 获取读热编码后的标签
            label = row.drop('images').drop('labels').astype(np.float32)
            self.__labels.append(label.values)

    # 数据集的大小
    def __len__(self): 
        return len(self.__labels)    

    # 获取数据集路径
    def get_imgs_path(self):
        return self.path

    # 获取图像和标签
    def __getitem__(self, idx): 
        img = Image.open(self.__imgs_path[idx])
        label = torch.tensor(self.__labels[idx], dtype=torch.float32)
        img = self.__transform(img) # 图像预处理

        return img, label

    # 独热编码
    def one_hot_encode(self, csv, labels_name, column='labels'):
        # 提取标签
        labels = []
        for label in csv[column]:
            img_label = []
            for category in label.split():
                img_label.append(labels_name.index(category))
            labels.append(img_label)

        # 独热编码
        new_df = pd.DataFrame(np.zeros((csv.shape[0], len(labels_name)), dtype=np.int8), columns=labels_name)
        for x, all_label in enumerate(labels):
            for y in all_label:
                new_df.iloc[x, y] = 1

        return pd.concat([csv, new_df], axis=1)

# 训练
class Trainer:
    def __init__(self, train_dataset, val_dataset, batch_size, num_workers, model, optimizer, loss_function, save_path):
        self.train_dataLoader = DataLoader(
            dataset = train_dataset, 
            batch_size = batch_size, 
            shuffle = True, 
            num_workers = num_workers,
            pin_memory = True
        )
        self.val_dataLoader = DataLoader(
            dataset = val_dataset,
            batch_size = batch_size, 
            shuffle = True, 
            num_workers = num_workers,
            pin_memory = True
        )

        self.save_path = save_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_workers = num_workers
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 如果有NVIDA显卡，转到GPU训练，否则用CPU
        self.tester = Tester(model, loss_function, optimizer)
        
        model.to(self.device)
        print(f"正在使用{self.device}")

    # 保存模型检查点
    def save_checkpoint(self, epoch, train_losses, val_losses, train_accuracies, val_accuracies):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
        model_name = self.model.get_name()
        save_path = f"{self.save_path}/{model_name}_epoch_{epoch}.pth"
        torch.save(state, save_path)
        print(f"模型已保存至{save_path}")

    # 加载模型检查点
    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            if self.device == "cpu":
                checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(checkpoint_path, weights_only=True)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"已加载模型{checkpoint_path}，epoch = {checkpoint['epoch']}")
            return checkpoint
        else:
            print(f"未找到模型{checkpoint_path}")
            return {}

    # 训练
    def train(self, num_epoch, checkpoint_path):
        # 初始化
        checkpoint = self.load_checkpoint(checkpoint_path) # 加载检查点
        batch_train_len = len(self.train_dataLoader) # batch的总数量
        batch_val_len = len(self.val_dataLoader) # batch的总数量
        start_epoch = checkpoint.get('epoch', 0) + 1
        train_losses = checkpoint.get('train_losses', []) # 训练损失
        val_losses = checkpoint.get('val_losses', []) # 验证损失
        train_accuracies = checkpoint.get('train_accuracies', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        len_trainset = len(self.train_dataset)
        len_valset = len(self.val_dataset)

        for epoch in range (start_epoch, num_epoch + 1):
            print(f"epoch: {epoch} / {num_epoch}", flush=True)
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            
            # 训练
            self.model.train()
            start_time = time.time()
            train_correct_num = 0 # 样本级
            for batch_idx, (x, y) in enumerate(self.train_dataLoader):
                if batch_idx % (batch_train_len / 10) == 0:
                    print(f"{batch_idx} / {batch_train_len}", end="\r",flush=True)

                x, y = x.to(self.device), y.to(self.device)

                # 前向传播
                output = self.model(x).to(self.device) # 模型输出
                batch_loss = self.loss_function(output, y).to(self.device) # 当前batch的损失
                epoch_train_loss += batch_loss.item()
                probs = torch.sigmoid(output) # Sigmoid激活后的概率
                train_correct_num += torch.all((probs > 0.5).int() == y, dim=1).sum().item()

                # 反向传播及优化
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            seconds = time.time() - start_time
            print(f"{int(seconds // 60)}分{seconds % 60:.2f}秒")

            # 平均训练误差和准确率
            avg_train_loss = epoch_train_loss / len_trainset
            train_acc = train_correct_num / len_trainset
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            print(f'训练集：平均误差 = {avg_train_loss} 准确率 = {train_acc*100:.6f}%')


            # 验证
            self.model.eval()
            start_time = time.time()
            val_correct_num = 0 # 样本级
            with torch.no_grad(): # 不计算梯度
                for batch_idx, (x, y) in enumerate(self.val_dataLoader):
                    if batch_idx % (batch_val_len / 10) == 0:
                        print(f"{batch_idx} / {batch_val_len}", end="\r",flush=True)
                        
                    x, y = x.to(self.device), y.to(self.device)

                    # 前向传播
                    output = self.model(x).to(self.device) # 模型输出
                    batch_loss = self.loss_function(output, y).to(self.device) # 当前batch的损失
                    epoch_val_loss += batch_loss.item()
                    probs = torch.sigmoid(output) # Sigmoid激活后的概率
                    val_correct_num += torch.all((probs > 0.5).int() == y, dim=1).sum().item()

            seconds = time.time() - start_time
            print(f"{int(seconds // 60)}分{seconds % 60:.2f}秒")

            # 平均训练误差和准确率
            avg_val_loss = epoch_val_loss / len_valset
            val_acc = val_correct_num / len_valset
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            print(f'验证集：平均误差 = {avg_val_loss} 准确率 = {val_acc*100:.6f}%')

            # 保存
            self.save_checkpoint(epoch, train_losses, val_losses, train_accuracies, val_accuracies)


# 测试
class Tester:
    def __init__(self, model, loss_function, threshold=0.5):
        self.threshold = threshold
        self.loss_function = loss_function
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 如果有NVIDA显卡，转到GPU训练，否则用CPU
        
        model.to(self.device)


    # 加载模型检查点
    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            if self.device == "cpu":
                checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(checkpoint_path, weights_only=True)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            print(f"已加载模型{checkpoint_path}，epoch = {checkpoint['epoch']}")
            return checkpoint
        else:
            print(f"未找到模型{checkpoint_path}")
            return {}

    # 测试
    def test(self, checkpoint_path, dataset, batch_size, num_workers):
        # 加载数据        
        dataLoader = DataLoader(
            dataset = dataset, 
            batch_size = batch_size, 
            shuffle = True, 
            num_workers = num_workers,
            pin_memory = True
        )

        # 初始化
        checkpoint = self.load_checkpoint(checkpoint_path)
        if len(checkpoint) == 0:
            return
        self.model.eval() # 设置为评估模式
        total_test_loss = 0.0
        test_correct_num = 0 # 预测正确的数量
        len_testset = len(dataset)


        epoch = checkpoint.get('epoch', 0)
        train_losses = checkpoint.get('train_losses', []) # 训练损失
        val_losses = checkpoint.get('val_losses', []) # 验证损失
        train_accuracies = checkpoint.get('train_accuracies', [])
        val_accuracies = checkpoint.get('val_accuracies', [])

        with torch.no_grad(): # 不计算梯度
            for batch_idx, (x, y) in enumerate(dataLoader):
                x, y = x.to(self.device), y.to(self.device)
    
                # 前向传播
                output = self.model(x).to(self.device) # 模型输出
                batch_loss = self.loss_function(output, y).to(self.device) # 当前batch的损失
                probs = torch.sigmoid(output) # Sigmoid激活后的概率
                correct = torch.all((probs > self.threshold).int() == y, dim=1).sum().item()
                test_correct_num += correct

                # 保存
                total_test_loss += batch_loss.item()
        
        # 平均误差
        avg_test_loss = total_test_loss / len_testset
        test_acc = test_correct_num / len_testset
        path = dataset.get_imgs_path()
        print(f"{path}：平均误差 = {avg_test_loss} 准确率 = {test_acc*100:.6f}%")


        plt.plot(range(epoch), train_losses, label="train")
        plt.plot(range(epoch), val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

        plt.scatter(range(epoch), train_accuracies, label="train")
        plt.scatter(range(epoch), val_accuracies, label="val")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()

# 预测一张图片
def predict_img(img_path, transform, model, model_path, labels_name, optimizer):
    # 加载模型
    checkpoint = Plant_dataset.load_checkpoint(model_path, model, optimizer)
    if len(checkpoint) == 0:
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval() # 设置模型为评估模式

    # 加载图片
    image = Image.open(img_path)

    # 获取模型输出
    with torch.no_grad():  # 不计算梯度
        img = transform(image).unsqueeze(0).to(device) # 预处理，增加batch维度
        logits = model(img).to(device)
    probs = torch.sigmoid(logits).flatten()
    
    for idx, probability in enumerate(probs):
        print(f"图片为{labels_name[idx]}的概率：{probability * 100:.12f}%")
    image.show()
