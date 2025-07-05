import os.path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models
import torch.optim as optim
from jinja2.optimizer import optimize
from PIL import Image
import SimpleITK as sitk
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
os.environ["SITK_SHOW_COMMAND"] = "F:\software\ImageJ\ImageJ.exe"
os.environ["SITK_SHOW_EXTENSION"] = ".nii"  # 使用NIfTI格式而不是默认的mha
from unet_model import Unet
from mri_data_set import MRIDataSet


class Train_Unet:
        def __init__(self,model,loss,opitimizer):
            self.model = model
            self.loss = loss
            self.optimizer = opitimizer
            self.writer = SummaryWriter('runs/experiment')
        def train_model(self,device,data_loader,n_epochs=10):
            self.model.to(device)
            self.model.train()
            scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,step_size=5,gamma=0.2)

            for epoch in tqdm(range(22,n_epochs)):
                tot_loss = 0.0
                for i,(inputs,labels) in enumerate(data_loader):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    #前向传播
                    outputs = self.model(inputs)
                    #loss
                    los = self.loss(outputs,labels)
                    #梯度清零
                    self.optimizer.zero_grad()
                    #
                    los.backward()
                    self.optimizer.step()
                    tot_loss += los
                    print(f"epoch={epoch},batch={i},tot_loss={tot_loss}")
                self.writer.add_scalar('loss/train',tot_loss,epoch)

                print("===============================================================")
                scheduler.step()

                print(f'epoch {epoch},loss={tot_loss},LR={scheduler.get_last_lr()[0]:.6f}')
                torch.save(self.model.state_dict(),f'model_state_dict_epoch{epoch}.mod')
        def predict_model(self,device,data_loader):
            self.model.to(device)
            self.model.eval()
            for i,(img,label) in enumerate(data_loader):
                with torch.no_grad():
                    img = img.to(device)
                    pred = self.model(img)
                    pred_mask = torch.argmax(torch.softmax(pred,1),1).squeeze(0)
                    s = torch.sum(label,dtype=torch.long)
                    if s.item() > 0:
                        plt.subplot(3,3,1)
                        plt.imshow(img.detach().clone().cpu().squeeze(0)[0])
                        plt.title("t1")
                        plt.subplot(3,3,2)
                        plt.imshow(img.detach().clone().cpu().squeeze(0)[1])
                        plt.title("t2")
                        plt.subplot(3, 3, 3)
                        plt.imshow(img.detach().clone().cpu().squeeze(0)[2])
                        plt.title("t1ce")
                        plt.subplot(3, 3, 4)
                        plt.imshow(img.detach().clone().cpu().squeeze(0)[3])
                        plt.title("flair")
                        plt.subplot(3, 3, 5)
                        plt.imshow(label.squeeze(0))
                        plt.title('label')
                        plt.subplot(3,3,6)
                        pred_mask = pred_mask.detach().clone().cpu()#torch.tensor(pred_mask).cpu()
                        plt.imshow(pred_mask)
                        plt.title('predicted')
                        plt.show(block = True)




if __name__ == '__main__':
    state = 'train'
    img_path = r"D:\python_code\MICCAI_BraTS_2019_Data_Training\HGG"
    data_set = MRIDataSet(img_path)
    data_loader = DataLoader(data_set,batch_size=5,shuffle=True)
    # fig = plt.figure(figsize=(5,5))
    model = Unet(4,5)
    loss = torch.nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('model_state_dict_epoch21.mod'))
    opt = optim.Adam(model.parameters(),lr=0.0001)
    # opt = optim.RMSprop(model.parameters(),lr=0.001,weight_decay=0.01,momentum=0.9)
    tr = Train_Unet(model,loss,opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if state == 'train':
        tr.train_model(device,data_loader,100)
    elif state == 'eval':
        data_loader1 = DataLoader(data_set, batch_size=1, shuffle=False)
        model.load_state_dict(torch.load('model_state_dict_epoch11.mod'))
        tr.predict_model(device, data_loader1)


