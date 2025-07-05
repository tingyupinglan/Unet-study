import matplotlib.pyplot as plt
import torch.nn
from tqdm.auto import tqdm
from fation_mnist_dataset import cls_mnist_fation_dataset
from unet_model import Unet_Classify
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
class MnistTrain:
    def __init__(self,batch_size=10,resize=None):
        ds = cls_mnist_fation_dataset()
        self.classes = 10
        self.in_channels = 1
        #获取数据
        self.train_data,self.test_data = ds.load_as_data_loader(batch_size,resize=resize,show_example=True)
        #设置模型
        self.model = Unet_Classify(self.in_channels,self.classes)
        #设置优化器
        self.optimizer = optim.RMSprop(self.model.parameters(),lr=0.001,weight_decay=1e-5,momentum=0.9)
        #设置loss
        self.loss_func = torch.nn.CrossEntropyLoss()
        #设置模型运行设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #定义优化可视化写入器
        self.visualize_writer = SummaryWriter(log_dir='runs/experiment/mnist_unet')
        #label的名字
        self.label_name = ds.get_label_name()
    def train_model(self,epochs=100):
        #设置为train模式
        self.model.train()
        self.model.to(self.device)
        min_loss = 1000000000000000.0
        #epoch进度条
        epoch_bar = tqdm(range(epochs),desc="训练进度",unit="epoch",position=0,
                         colour='green',bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for epoch in epoch_bar:
            tot_loss = 0.0
            # 更新 epoch 描述
            epoch_bar.set_description(f"Epoch {epoch + 1}/{epochs}")

            batch_bar = tqdm(enumerate(self.train_data),total=len(self.train_data),unit="batch",position=1,leave=False,dynamic_ncols=True,
                             colour='red',
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')
            for i,(imgs,labels) in batch_bar:
                #把输入图像和labels放入设备，gpu或者cpu
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                #前向计算
                output_imgs = self.model(imgs)
                #计算loss
                loss = self.loss_func(output_imgs,labels)
                #梯度清零
                self.optimizer.zero_grad()
                #反向计算梯度
                loss.backward()
                #执行一步梯度反向
                self.optimizer.step()
                tot_loss += loss

                # batch_bar.set_postfix({'tot_loss':f'{tot_loss.item():.4f}'},refresh=True)
                sys.stdout.flush()
                desc = f"Batch {i}/{len(self.train_data)} | Loss: {tot_loss.item():.4f}"
                batch_bar.set_description(desc)
            batch_bar.close()
            avg_loss = tot_loss / len(self.train_data)

            torch.save(self.model.state_dict(),f'mnist_model_para/model_para_epochs{epoch}.mod')
            self.visualize_writer.add_scalar('avg_loss',avg_loss,global_step=epoch)
            #print(f'\n-----epoch={epoch},tot_loss={tot_loss}-----')
            if tot_loss < min_loss:
                torch.save(self.model.state_dict(),f'mnist_model_para/best_model_para.mod')
    def predict_model(self,model_para_file,show_img=False):
        self.model.load_state_dict(torch.load(model_para_file))
        self.model.eval()
        self.model.to(self.device)
        n_correct = 0
        n_total = 0
        test_bar = tqdm(enumerate(self.test_data),position=0,bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]',colour='red')
        for i,(imgs,labels) in test_bar:
            test_bar.set_description(f'{i}/{len(self.test_data)}')
            imgs = imgs.to(self.device)
            pred_class = self.model(imgs)
            cls_idx = torch.argmax(torch.softmax(pred_class,dim=1)).detach().cpu()
            n_correct = (cls_idx == labels).sum().item()
            n_total = labels.size(0)
            if show_img:
                imgs = imgs.detach().cpu()
                for n in range(labels.size(0)):
                    plt.imshow(imgs[n][0])
                    plt.title(f'label={self.label_name.get(labels[n].item())},predict={self.label_name.get(cls_idx.item())}')
                    plt.show(block=True)

        accuracy = n_correct / n_total * 100
        print(f'accuracy={accuracy:.4f}%')




if __name__ == '__main__':
    mode = 'eval'
    if mode == 'train':
        train_model = MnistTrain(batch_size=30)
        train_model.train_model(epochs=100)
    else:
        eval_model = MnistTrain(batch_size=1)
        file_para = 'mnist_model_para/best_model_para.mod'
        eval_model.predict_model(show_img=False,model_para_file=file_para)