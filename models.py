from torch import nn
import torch

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        # 使用sigmoid作为激活函数
        self.Sigmoid = nn.Sigmoid()

        # 卷积层，输入大小为28*28，输出大小为28*28，输入通道为3（RGB三个通道），输出为6，卷积核为5，扩充边缘为2
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)

        # AvgPool2d：二维平均池化操作
        # 池化层，输入大小为28*28，输出大小为14*14，输入通道为6，输出为6，卷积核为2，步长为2
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 卷积层，输入大小为14*14，输出大小为10*10，输入通道为6，输出为16，卷积核为5
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # 池化层，输入大小为10*10，输出大小为5*5，输入通道为16，输出为16，卷积核为2，步长为2
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 卷积层，输入大小为5*5，输出大小为1*1，输入通道为16，输出为120，卷积核为5
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        
        # 全连接层
        # Flatten()：将张量（多维数组）平坦化处理，张量的第0维表示的是batch_size（数量），所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()
        # Linear（in_features，out_features）
        # in_features指的是[batch_size, size]中的size,即样本的大小
        # out_features指的是[batch_size，output_size]中的output_size，样本输出的维度大小，也代表了该全连接层的神经元个数
        self.f6 = nn.Linear(300000, 84)
        
        # 输出层
        self.output = nn.Linear(84, num_classes)

        # 如果有NVIDA显卡，转到GPU训练，否则用CPU
        self.to('cuda' if torch.cuda.is_available() else 'cpu') 
 
 
    def forward(self, x):
        x = self.Sigmoid(self.c1(x)) # 卷积
        x = self.s2(x) # 池化
        x = self.Sigmoid(self.c3(x)) # 卷积
        x = self.s4(x) # 池化
        x = self.c5(x) # 卷积
        x = self.f6(self.flatten(x)) # 全连接
        x = self.output(x)  # 输出
        return x