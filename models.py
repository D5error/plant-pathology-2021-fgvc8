from torch import nn
import torch

class LeNet5(nn.Module):
    def __init__(self, num_classes=6):
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

    def forward(self, x):
        x = self.Sigmoid(self.c1(x)) # 卷积
        x = self.s2(x) # 池化
        x = self.Sigmoid(self.c3(x)) # 卷积
        x = self.s4(x) # 池化
        x = self.c5(x) # 卷积
        x = self.f6(self.flatten(x)) # 全连接
        x = self.output(x)  # 输出
        return x
    
    def get_name(self):
        return "LeNet5"
    

class Vgg(nn.Module):  # VGG继承nn.Module(所有神经网络的基类)
    Vgg11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    Vgg13 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    Vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    Vgg19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

    def __init__(self, features, name, num_classes=6, init_weights=False, weights_path=None):
        """生成的网络特征，分类的个数，是否初始化权重，权重初始化路径"""
        super(Vgg, self).__init__()  # 多继承
        self.features = features
        self.name = name
        self.classifier = nn.Sequential(
            # 将最后三层全连接和分类进行打包
            # torch.nn.Squential：一个连续的容器。模块将按照在构造函数中传递的顺序添加到模块中。或者，也可以传递模块的有序字典。
            nn.Dropout(p=0.5),  # 随机失活一部分神经元，用于减少过拟合，默认比例为0.5，仅用于正向传播。
            nn.Linear(512*7*7, 2048),
            # 全连接层，将输入的数据展开成为一位数组。将512*7*7=25088展平得到的一维向量的个数为2048
            nn.ReLU(True),  # 定义激活函数
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),  # 定义第二个全连接层
            nn.ReLU(True),
            nn.Linear(2048, num_classes)  # 最后一个全连接层。num_classes：分类的类别个数。
        )
        if init_weights and weights_path is None:
            self._initialize_weights()  # 是否对网络进行初始化

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path), strict=False)

    def forward(self, x):
        """
        前向传播,x是input进来的图像
        """
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)  # 进行展平处理，start_dim=1，指定从哪个维度开始展平处理，
        # 因为第一个维度是batch，不需要对它进行展开，所以从第二个维度进行展开。
        # N x 512*7*7
        x = self.classifier(x)  # 展平后将特征矩阵输入到事先定义好的分类网络结构中。
        return x
    
    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():  # 用m遍历网络的每一个子模块，即网络的每一层
            if isinstance(m, nn.Conv2d):  # 若m的值为 nn.Conv2d,即卷积层
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  # 用xavier初始化方法初始化卷积核的权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 若偏置不为None，则将偏置初始化为0
            elif isinstance(m, nn.Linear):  # 若m的值为nn.Linear,即池化层
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                # 用一个正太分布来给权重进行赋值，0为均值，0.01为方差
                nn.init.constant_(m.bias, 0)

    def make_features(cfg: list, ):
        """
        提取特征网络结构，
        cfg.list：传入配置变量，只需要传入对应配置的列表
        """
        layers = []  # 空列表，用来存放所创建的每一层结构
        in_channels = 3  # 输入数据的深度，RGB图像深度数为3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                # 若为最大池化层，创建池化操作，并为卷积核大小设置为2，步距设置为2，并将其加入到layers
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(True)]
                in_channels = v
                # 创建卷积操作，定义输入深度，配置变量，卷积核大小为3，padding操作为1，并将Conv2d和ReLU激活函数加入到layers列表中
        return nn.Sequential(*layers)

    def get_name(self):
        return self.name

# 选择模型
def select_model(name):
    if "LeNet5" == name:
        return LeNet5()
    elif "Vgg11" == name:
        cfg = Vgg.Vgg11
        return Vgg(Vgg.make_features(cfg), name)
    elif "Vgg13" == name:
        cfg = Vgg.Vgg13
        return Vgg(Vgg.make_features(cfg), name)
    elif "Vgg16" == name:
        cfg = Vgg.Vgg16
        return Vgg(Vgg.make_features(cfg), name)
    elif "Vgg19" == name:
        cfg = Vgg.Vgg19
        return Vgg(Vgg.make_features(cfg), name)
    else:
        return None