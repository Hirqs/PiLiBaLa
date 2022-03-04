import os
import sys
import json
from datetime import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet



def main():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {# 这是一个字典 存放了train 和 val的transform方式
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪，裁剪到224x224大小
                                     transforms.RandomHorizontalFlip(),  # 随机翻转，数据增强一种方法，这里是水平翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # 标准化处理，防止突出数值较高的指标在综合分析中的作用
        "val": transforms.Compose([transforms.Resize((224, 224)),   # cannot 224, must (224, 224) 强制转化为224x224
                                   transforms.ToTensor(),           # 转化为张量
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # os.getcwd() 获取当前所在文件的目录
    # data_root = os.path.abspath(）获取数据集所在根目录
    # os.path.join() 将传入两个路径连接在一起
    # "../.." 表示返回上上一层目录
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))                 # ..代表返回上层目录，即PiLiPaLa目录下
    image_path = os.path.join(data_root, "data_set", "flower_data")                 # PiLiPaLa/data_set/flower_data
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path) # assert 如果报错 在控制台输入后面的字符串信息
    # datasets.ImageFolder() 这个函数加载数据集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),    # 训练集的根目录 PiLiPaLa/data_set/flower_data/train
                                         transform=data_transform["train"]          # 用train所需要的transform
                                         )
    train_num = len(train_dataset)      # 获取训练集长度
    print('train_num is ', train_num)   # 3306个训练样本


    # classes, class_to_idx = self.find_classes(self.root)
    # classes是一个列表 ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    # class_to_idx是一个字典(key,value) 赋值给了 flower_list ，{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    flower_list = train_dataset.class_to_idx                        # 获取分类名称对应索引 通过前面几行代码设置的断点可以追到源码
    # cla_dict = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    cla_dict = dict((val, key) for key, val in flower_list.items()) # 将key和value对调
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)                       # json.dumps将一个Python数据结构转换为JSON 解析：https://blog.csdn.net/weixin_38842821/article/details/108359551
    with open('class_indices.json', 'w') as json_file:              # 保存到一个json文件当中
        json_file.write(json_str)

    batch_size = 32
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw=1
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  # batch_size=4, shuffle=True,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # 查看数据集 需要把shuffle设置True，见上一个注释
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())                         # 查看模型的一个参数
    optimizer = optim.Adam(net.parameters(), lr=0.0002)     # 学习率自己调整

    epochs = 10
    save_path = './AlexNet.pth'  # ./代表同级目录
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()                 # 使用了dropout 但是只希望在训练中随机失活 在预测过程中不希望它起作用 net.train()开启dropout  net.eval()关闭dropout
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)         # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        # t1 = time.perf_counter()                                # 训练一个epoch所需时间
        for step, data in enumerate(train_bar):                 # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        # for step, data in enumerate(train_loader, start=0):
            images, labels = data                               # 在训练集的加载中 datasets.ImageFolder 追进源码可知，__getitem__返回的是样本和标签，所以这里用images, labels接收data数据
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))    # 此处的loss是一个tensor张量 <class 'torch.Tensor'>
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()                         # pytorch中的.item()用于将一个零维张量转换成浮点数

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():                                   # 禁止参数跟踪
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))


                '''
                dim 是max函数 索引 的维度,0是按列，1是按行。-1是最后一个维度，一般是按行。
                由于验证集的batch_size为4，一共有五个类别，所以没遍历一次就输出一个tensor(4,5)
                tensor([[-0.2434,  0.5892, -1.1177,  0.9521, -0.4927],
                        [ 1.0084,  0.5093, -0.4579, -0.8835, -0.3898],
                        [ 0.4746,  0.1932, -0.1672, -0.4958, -0.1249],
                        [ 0.7939,  0.5353, -0.5026, -0.6986, -0.3449]], device='cuda:0')
                '''
                predict_y = torch.max(outputs, dim=1)[1]
                print('-*-*-*-*-*-*-*-*-*-*-*-*-**-*--*-')
                print(predict_y)
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # 查eq是什么意思 以及继续后面的操作  打开桌面的网址

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
