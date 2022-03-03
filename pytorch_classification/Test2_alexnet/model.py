# AlexNet 亮点
# 1. 首次利用GPU进行网络加速
# 2.使用了Relu激活函数，而不是传统的sigmoid激活以及tanh激活
# 3.使用了LRN局部响应归一化
# 4.在全连接层的前两层使用了Dropout 随机失活神经元操作，以减少过拟合 重点!!!

import torch.nn as nn
import torch
# 网络结构分析
# conv1：rgb kernels：48 x 2 = 96  kernel_size = 11 padding[1, 2] stride = 4
# input_size [224, 224, 3] output_size [55, 55, 96]
# maxpool1: kernel_size:3 pading:0 stride:2 output_size[27, 27, 96]
# conv2: kernels: 128 x 2 = 256 kernel_size: 5 padding[2, 2] stide = 1
# output_size [27, 27, 256]
# maxpool2: kernel_size = 3 padding=0 stride=2 output_size[13, 13, 256]
# conv3: kernels: 192 x 2 = 384 kernel_size= 3 padding:[1, 1] stride:1  output_size[13, 13, 384]
# conv4:kernels: 384 kernel_size: 3 padding:[1,1] stride:1  output_size[13,13,384]   这里可以看到输出size并没有变化 但这个网络仍然加入了这个卷积层
# conv5:kernels: 256 kernel_size: 3 padding:[1,1] stride:1  output_size[13,13,256]
# maxpool3: kernel_size:3 padding:0 stride:2    output_size[6, 6, 256]
# 三个全连接层 最后一个分类是1000


# 公式复习 N=(W-F+2P)/S+1
# 输入图片的大小wXw 卷积核大小FxF 步长S padding p

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):   # init_weight 初始化权重
        super(AlexNet, self).__init__()
        self.features = nn.Sequential( # nn.Sequential 将一系列层结构打包 .features 专门用于提取图像的特征的结构
            # 原论文中卷积核个数是96个 这里由于数据集较小 将其减半 最后效果差不多
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] padding只能传入两种形式的变量 一种是int 一种是tuple
            #  如果传入tuple 如果是[1, 2]  上下补一行0 左右补两行0
            nn.ReLU(inplace=True),                                  # inplace 增加计算量但是降低内存使用的一种方法
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(            # 分类器 classifier
            nn.Dropout(p=0.5),                      # 随机失活比例 默认0.5
            nn.Linear(128 * 6 * 6, 2048),           # 2048全连接层节点个数
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),           # num_classes 初始化类别 原本有1000 但本数据集中只用了5
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)           # flatten 将传入的变量展平 从channel维度开始展平 因为第一维是batch 和LeNet中相同
        x = self.classifier(x)                      # 输入到分类结构
        return x

    def _initialize_weights(self):
        for m in self.modules():                    # 继承自nn.Module 迭代定义的每一层结构 遍历了每一层结构之后判断他属于哪一个类别
            if isinstance(m, nn.Conv2d):            # 遍历了每一层结构之后判断他属于哪一个类别
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 如果他是卷积层 就用kaiming_normal_这个方法去初始化权重
                if m.bias is not None:                                                  # 如果权重不是空值就清空为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):                                              # 如果是线性层就用init.normal_来初始化
                nn.init.normal_(m.weight, 0, 0.01)                                      # 用正态分布 均值为0， 方差为0.01
                nn.init.constant_(m.bias, 0)                                            # 同样初始化bias为0 实际上自动使用kaiming初始化
