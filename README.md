I create this project to study unet, i try to code the unet and train all kinds of image tasks.
我创建这个项目是为了学习Unet网络，我自己手动写unet网络，并执行各种图像训练任务
fation_mnist_dataset.py:自动下载fashion mnist图像数据，并返回pytorch的dataloader
mnist_fation_train.py：使用unet训练mnist fashion数据
unet_model.py：unet网络模型定义
unet_train_mri_data_set.py：训练来自kaggle的MRI图像数据
mri_data_set.py: 读取来自kaggel的MRI图像数据
unet_model_3D.py：为mri的3D数据写的unet模型
unet_train_3D.py：为mri的3D数据写的训练过程，需要更多的显存和内存
