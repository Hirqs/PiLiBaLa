import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "../../data_set/flower_data/val/tulips/38287568_627de6ca20.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)               # torch.Size([3, 224, 224])
    # torch.unsqueeze()这个函数主要是对数据维度进行扩充。第二个参数为0数据为行方向扩充，为1列方向扩充
    img = torch.unsqueeze(img, dim=0)       # torch.Size([1, 3, 224, 224])
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))             # model.state_dict()返回一个OrderDict，存储了网络结构的名字和对应的参数


    model.eval()
    with torch.no_grad():                                       # 被with torch.no_grad()包住的代码，不用跟踪反向梯度计算
        # predict class
        # squeeze的用法主要是对数据的维度进行压缩,去掉维数为1的的维度 比如是一行或者一列这种 一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        # torch.argmax()    不指定dim，返回张量中所有数据的最大值位置（按照张量被拉伸为一维向量算）；
        #                   指定dim，返回指定维度的最大值位置，另外可以通过keepdim来保留原张量的形状
        # torch.max(),      不指定dim参数，返回输入张量中所有数据的最大值；
        #                   指定dim参数，则返回按照指定维度的最大值和各最大值对应的位置
        predict_cla = torch.argmax(predict).numpy()


    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
