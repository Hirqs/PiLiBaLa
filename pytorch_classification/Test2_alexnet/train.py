import os
import sys
import json

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
    cla_dict = dict((val, key) for key, val in flower_list.items()) # 循环遍历数组索引核值并交换重新赋值给数组，这样模型预测出来的直接就是value类别值
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
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
    # 查看数据集
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
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './AlexNet.pth'  # ./代表同级目录
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
