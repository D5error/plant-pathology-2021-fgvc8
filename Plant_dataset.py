# -*- coding: utf-8 -*-
import os
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import LabelEncoder
from torch.utils.data import DataLoader
import torch


class Plant_dataset: 
    def __init__(self, dataset_path, train_or_test, checkpoint_path, transform=None):
        self.checkpoint_path = checkpoint_path # 模型保存的路径
        self.dataset_path = dataset_path # 数据集的路径
        self.transform = transform # 预处理（数据增强等）
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 如果有NVIDA显卡，转到GPU训练，否则用CPU
        print(f"正在使用{self.device}")
        self.encoder = LabelEncoder() # 使用 LabelEncoder 进行标签编码
        # 解编码：encoder.inverse_transform(y_encoded)

        # 读取数据集
        csv = pd.read_csv(self.dataset_path + "/" + train_or_test + "/"  + train_or_test + '_label.csv')
        
        # 获取图像路径和标签
        self.images_path, self.labels = [], []
        for _, row in csv.iterrows():
            image_path = os.path.join(self.dataset_path + "/" + train_or_test + '/images', row['images'])
            self.images_path.append(image_path)
            self.labels.append(row['labels'])
        self.labels = self.encoder.fit_transform(self.labels) # 将标签编码


    def __len__(self): # 获取标签长度
        return len(self.labels)    


    def __getitem__(self, idx): # 获取图像的标签
        img = Image.open(self.images_path[idx])
        label = self.labels[idx]
        # 图像预处理
        if self.transform: 
            img = self.transform(img)

        return img, label


    def test(self, dataloader, model, loss_function):
        model.eval() # 设置为评估模式

        total_loss, total_acc = 0.0, 0.0

        with torch.no_grad(): # 不计算梯度
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                batch_loss = loss_function(output, y)
                total_loss += batch_loss.item()
                _, pred = torch.max(output, axis=1)
                correct = torch.sum(pred == y).item()
                batch_acc = correct / output.shape[0]
                total_acc += batch_acc
                print(f"\rbatch_index: {batch_idx} / {len(dataloader)-1}", end="")

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        print(f"\rtest avg_loss: {avg_loss}")
        print(f"test avg_acc: {100 * avg_acc:.4f}%")


    def train(self, num_epoch, dataLoader, model, loss_function, optimizer):
        total_loss, total_acc = [], []

        for epoch in range (0, num_epoch): 
            print(f"epoch: {epoch}")
            epoch_loss, epoch_acc = 0.0, 0.0
            
            for batch_idx, (x, y) in enumerate(dataLoader):
                print(f"\rbatch_index: {batch_idx} / {len(dataLoader) - 1}", end="")
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                batch_loss = loss_function(output, y)
                epoch_loss += batch_loss.item()
                # torch.max(input, dim)函数
                # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
                # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
                _, pred = torch.max(output, axis=1)
                correct = torch.sum(pred == y).item()
                # 计算每批次的准确率
                # output.shape[0]一维长度为该批次的数量
                # torch.sum()对输入的tensor数据的某一维度求和
                batch_acc = correct / output.shape[0]
                epoch_acc += batch_acc
                # 反向传播及优化
                # 清空过往梯度
                optimizer.zero_grad()
                # 反向传播，计算当前梯度
                batch_loss.backward()
                # 根据梯度更新网络参数
                optimizer.step()

            print(f"\rtotal epoch: {num_epoch}")
            avg_loss = epoch_loss / len(dataLoader)
            total_loss.append(avg_loss)
            print('train avg_loss:',  avg_loss)

            avg_acc = epoch_acc / len(dataLoader)
            total_acc.append(avg_acc)
            print(f'train avg_acc: {100 * avg_acc:.2f}%')

            '''
            这里加点可视化，如误差曲线，ROC等等
            '''
            plt.subplot(121)
            plt.plot(range(num_epoch), total_loss)
            plt.subplot(122)
            plt.plot(range(num_epoch), total_acc)
            plt.show()



# 运行
def start(dataset_path, transform, train_or_test, checkpoint_path, batch_size, num_workers, model, loss_function, optimizer, num_epoch):
    dataset = Plant_dataset(dataset_path, train_or_test, checkpoint_path, transform) # 加载数据集
    dataLoader = DataLoader(dataset, batch_size, True, num_workers = num_workers)

    if train_or_test == "train": # 训练
        dataset.train(num_epoch, dataLoader, model, loss_function, optimizer)
        torch.save(model.state_dict(), checkpoint_path)
    elif train_or_test == "test": # 测试
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        dataset.test(dataLoader, model, loss_function)