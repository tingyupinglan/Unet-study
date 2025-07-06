import os.path

import numpy as np
import torch
import torchvision.models
import torch.optim as optim
from jinja2.optimizer import optimize
from matplotlib.pyplot import subplot
from torch.utils.data import Dataset
from PIL import Image
import SimpleITK as sitk
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
os.environ["SITK_SHOW_COMMAND"] = "F:\software\ImageJ\ImageJ.exe"
os.environ["SITK_SHOW_EXTENSION"] = ".nii"  # 使用NIfTI格式而不是默认的mha
from unet_model_3D import Unet
# from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('TkAgg')    # 通常最可靠

class MyDataSet(Dataset):
    def __init__(self,image_dir,image_transform=None,mask_transform=None):
        self.img_dir = image_dir
        # self.mask_dir = mask_dir

        self.img_path = [os.path.join(self.img_dir,f) for f in os.listdir(self.img_dir)]
        # self.mask_path = [os.path.join(self.mask_dir,f) for f in os.listdir(self.mask_dir)]
        self.img_transform = image_transform
        self.mask_transform = mask_transform
        # #确保mask和img对应
        # self.img_path.sort()
        # self.mask_path.sort()

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        file_path = self.img_path[item]
        file_t1 = self.find_file_with_str(file_path,"_t1.")

        if len(file_t1) != 1:
            raise ValueError(f"{file_path}中没有t1文件或者t1文件数大于1")
        img_t1 = sitk.ReadImage(os.path.join(file_path,file_t1[0]))
        # sitk.Show(img_t1,title="img")
        img_all = sitk.GetArrayFromImage(img_t1)

        # idx_rnd = torch.randint(0,img_all.shape[0]-1,(1,))
        #三维图像
        img = img_all[:,:,:].astype(np.float32)

        file_seg = self.find_file_with_str(file_path,"_seg.")
        if len(file_seg) != 1:
            raise ValueError(f"{file_path}中没有mask文件或者mask文件数大于1")
        msk_seg = sitk.ReadImage(os.path.join(file_path,file_seg[0]))
        mask_all = sitk.GetArrayFromImage(msk_seg)
        mask = torch.tensor(mask_all[:,:,:],dtype=torch.long)
        # torch.randint(0,)
        # img = Image.open(self.img_path[item]).convert('RGB')
        # mask = Image.open(self.mask_path[item]).convert('L')
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        img = torch.unsqueeze(torch.tensor(img),0)

        # mask = torch.unsqueeze(torch.tensor(mask),0)

        return img,mask

    def find_file_with_str(self,path,str):
        files = os.listdir(path)
        file_list = []
        for file in files:
            if os.path.isfile(os.path.join(path,file)) and str in file:
                file_list.append(file)
        return file_list


class Train_Unet:
        def __init__(self,model,loss,opitimizer):
            self.model = model
            self.loss = loss
            self.optimizer = opitimizer
        def train_model(self,device,data_loader,n_epochs=10):
            self.model.to(device)
            self.model.train()
            tot_los = 0.0
            for epoch in range(n_epochs):
                tot_los = 0.0
                for i, (inputs, labels) in enumerate(data_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 前向传播
                    outputs = self.model(inputs)
                    # if not outputs.requires_grad():
                    #     outputs.requires_grad(True)
                    # loss
                    los = self.loss(outputs, labels)
                    # 梯度清零
                    self.optimizer.zero_grad()
                    #
                    los.backward()
                    self.optimizer.step()
                    tot_los += los
                print(f'epoch={epoch},loss={tot_los}')
            torch.save(self.model.state_dict(),'model_state_dict.mod')

        def model_predict(self,device,data_loader,n_eval):
            self.model.eval()
            self.model.to(device)

            for ev in range(n_eval):
                for i, (img, label) in enumerate(data_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        ev_mask = self.model(img)
                        mm = torch.argmax(torch.softmax(ev_mask, 1), 1).squeeze(0)
                        lbl_mask = label
                        plt.ion()  # 开启交互模式
                        plt.subplot(1, 2, 1)
                        plt.imshow(lbl_mask.squeeze(0))
                        plt.title('label')
                        plt.subplot(1, 2, 2)
                        plt.imshow(mm.squeeze(0))
                        plt.title('predict')
                        plt.show(block=True)
                        aa = 1


if __name__ == '__main__':
    img_path = r"D:\python_code\MICCAI_BraTS_2019_Data_Training\HGG"

    data_set = MyDataSet(img_path)
    data_loader = DataLoader(data_set,batch_size=1,shuffle=True)
    # fig = plt.figure(figsize=(5,5))
    model = Unet(1,5)
    # model.load_state_dict(torch.load('model_state_dict.mod'))
    loss = torch.nn.CrossEntropyLoss()

    opt = optim.Adam(model.parameters(),lr=0.001)
    tr = Train_Unet(model,loss,opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr.train_model(device,data_loader,20)
    # data_loader1 = DataLoader(data_set, batch_size=1, shuffle=False)
    # tr.model_predict(device,data_loader1,10)