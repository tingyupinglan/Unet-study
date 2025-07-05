import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as torch_data_sets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Subset
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class cls_mnist_fation_dataset(Dataset):
    def __init__(self):
        self.label_name = { 0:'T-shirt/top_T恤/上衣',
                            1:'Trouser:裤子',
                            2:'Pullover:套头衫（如毛衣）',
                            3:'Dress:连衣裙',
                            4:'Coat:外套',
                            5:'Sandal:凉鞋',
                            6:'Shirt:衬衫',
                            7:'Sneaker:运动鞋',
                            8:'Bag：包',
                            9:'AnkleBoot：短靴'
                          }

    def load_as_data_loader(self,batch_size=10,resize=None,show_example=False):
        #定义变换，意思是把数据转成tensor
        if resize is not None:
            trans_list = [transforms.Resize(14), transforms.ToTensor]
        else:
            trans_list = [transforms.ToTensor()]

        trans = transforms.Compose(trans_list)
        mnist_train = torch_data_sets.FashionMNIST(root='mnist_fation_data/train',train=True,transform=trans,download=True)
        # subset_train = Subset(mnist_train,range(100))
        mnist_test = torch_data_sets.FashionMNIST(root='mnist_fation_data/test',train=False,transform=trans,download=True)
        # print(f'size={len(mnist_test)},shape={mnist_train[0][0].shape}')
        print(len(mnist_train))
        if show_example:
            img_ex = mnist_train[0][0]
            label_ex = mnist_test[0][1]
            self.show_img_and_label(img_ex,label_ex)
        return DataLoader(mnist_train,shuffle=True,batch_size=batch_size),DataLoader(mnist_test,shuffle=False,batch_size=batch_size)

    def get_label_name(self):
        return self.label_name

    def show_img_and_label(self,img,label):
        plt.imshow(img.squeeze(0))
        plt.title(f'{self.label_name.get(label)}')
        plt.show(block=True)




