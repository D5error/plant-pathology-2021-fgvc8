# -*- coding: utf-8 -*-
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch


class Plant_dataset: 
    def __init__(self, dataset_path, train_or_test, checkpoint_path, transform, labels_name, threshold):
        # 保存参数
        self.checkpoint_path = checkpoint_path # 模型保存的路径
        self.dataset_path = dataset_path # 数据集的路径
        self.transform = transform # 预处理（数据增强等）
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 如果有NVIDA显卡，转到GPU训练，否则用CPU
        self.labels_name = labels_name # 标签名称
        self.threshold = threshold # 分类阈值
        print(f"正在使用{self.device}")

        # 读取数据集
        csv = pd.read_csv(self.dataset_path + "/" + train_or_test + "/"  + train_or_test + '_label.csv')
        csv = one_hot_encode(csv, labels_name) # 独热编码

        # 获取图像路径和标签
        self.images_path, self.labels = [], []
        for _, row in csv.iterrows():
            image_path = os.path.join(self.dataset_path + "/" + train_or_test + '/images', row['images'])
            self.images_path.append(image_path)

            # 获取读热编码后的标签
            label = row.drop('images').drop('labels')
            label = label.astype(np.float32) # 将标签转换为纯数字类型（确保是浮点型或整型）
            self.labels.append(label.values)

    # 数据集的大小
    def __len__(self): 
        return len(self.labels)    

    # 获取图像和标签
    def __getitem__(self, idx): 
        img = Image.open(self.images_path[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # 图像预处理
        img = self.transform(img)

        return img, label

    # 测试
    def test(self, dataLoader, model, loss_function):
        model.eval() # 设置为评估模式

        total_loss, total_acc = 0.0, 0.0

        with torch.no_grad(): # 不计算梯度
            for batch_idx, (x, y) in enumerate(dataLoader):
                print(f"\rbatch_index: {batch_idx} / {len(dataLoader) - 1}", end="")
                x, y = x.to(self.device), y.to(self.device)
    
                # 核心部分
                output = model(x) # 模型输出
                batch_loss = loss_function(output, y) # 当前batch误差
                preds = torch.sigmoid(output) # 激活函数归一化
                pred_labels = (preds > self.threshold).float() # 概率大于阈值预测为1
                correct = torch.sum(pred_labels == y).item() # 计算匹配的标签数
                batch_acc = correct / y.numel() # 当前batch准确率

                # 保存
                total_loss += batch_loss.item()
                total_acc += batch_acc
        
        # 平均误差
        avg_loss = total_loss / len(dataLoader)
        print(f"\n平均测试误差: {avg_loss}")

        # 平均准确率
        avg_acc = total_acc / len(dataLoader)
        print(f"平均测试准确率: {100 * avg_acc:.4f}%")

    # 训练
    def train(self, num_epoch, dataLoader, model, loss_function, optimizer):
        total_loss, total_acc = [], []

        for epoch in range (0, num_epoch): 
            print(f"\nepoch: {epoch + 1} / {num_epoch}")
            epoch_loss, epoch_acc = 0.0, 0.0
            
            for batch_idx, (x, y) in enumerate(dataLoader):
                print(f"\rbatch_index: {batch_idx} / {len(dataLoader) - 1}", end="")
                x, y = x.to(self.device), y.to(self.device)

                # 核心部分
                output = model(x) # 模型输出
                batch_loss = loss_function(output, y) # 当前batch误差
                preds = torch.sigmoid(output) # 激活函数归一化
                pred_labels = (preds > self.threshold).float() # 概率大于阈值预测为1
                correct = torch.sum(pred_labels == y).item() # 计算匹配的标签数
                batch_acc = correct / y.numel() # 当前batch准确率

                # 保存
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc

                # 反向传播及优化
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # 平均误差
            avg_loss = epoch_loss / len(dataLoader)
            total_loss.append(avg_loss)
            print(f'\n平均训练误差: {avg_loss}')

            # 平均准确率
            avg_acc = epoch_acc / len(dataLoader)
            total_acc.append(avg_acc)
            print(f'平均训练准确率: {100 * avg_acc:.2f}%')

        '''
        这里加点可视化，如误差曲线，ROC等等
        '''
        plt.subplot(121)
        plt.plot(range(num_epoch), total_loss)
        plt.subplot(122)
        plt.plot(range(num_epoch), total_acc)
        plt.show()


# 定义独热编码函数
def one_hot_encode(data, labels_name, column='labels'):
    # 提取标签
    labels = []
    for label in data[column]:
        all_label = []
        for category in label.split():
            all_label.append(labels_name.index(category))
        labels.append(all_label)

    # 独热编码
    new_df = pd.DataFrame(np.zeros((data.shape[0], len(labels_name)), dtype=np.int8), columns=labels_name)
    for x, all_label in enumerate(labels):
        for y in all_label:
            new_df.iloc[x, y] = 1

    return pd.concat([data, new_df], axis=1)

# 运行
def start(dataset_path, transform, train_or_test, checkpoint_path, batch_size, num_workers, model, loss_function, optimizer, num_epoch, labels_name, threshold):
    dataset = Plant_dataset(dataset_path, train_or_test, checkpoint_path, transform, labels_name, threshold) # 加载数据集
    dataLoader = DataLoader(dataset, batch_size, True, num_workers = num_workers)

    if train_or_test == "train": # 训练
        dataset.train(num_epoch, dataLoader, model, loss_function, optimizer)
        torch.save(model.state_dict(), checkpoint_path)
    elif train_or_test == "test": # 测试
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        dataset.test(dataLoader, model, loss_function)


# 预测一张图片
def predict_img(img_path, transform, model, num_classes, model_path, labels_name):
    # 加载模型
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() # 设置模型为评估模式

    # 加载图片
    image = Image.open(img_path)
    img = transform(image).unsqueeze(0) # 预处理，增加batch维度

    # 获取模型输出
    with torch.no_grad():  # 不计算梯度
        output = model(img)  

    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    for idx, probability in enumerate(probabilities):
        print(f"图片为{labels_name[idx]}的概率：{probability * 100:.2f}%")
    image.show()

# 测试bug用
if __name__ == "__main__":
    labels_name = [  # 标签
        "scab", # 癣病
        "healthy", # 健康
        "frog_eye_leaf_spot", # 蛙眼叶斑病
        "rust", # 锈病
        "complex", # 复杂病
        "powdery_mildew" # 白粉病
    ]
    data = pd.read_csv("./plant_dataset/test/test_label.csv")
    data = one_hot_encode(data, labels_name)
    print(data)