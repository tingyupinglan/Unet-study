from torch.utils.data import Dataset
class MRIDataSet(Dataset):
    def __init__(self,image_dir,image_transform=None,mask_transform=None):
        self.img_dir = image_dir
        self.n_vol_size = 155

        self.img_path = [os.path.join(self.img_dir,f) for f in os.listdir(self.img_dir)]
        # self.mask_path = [os.path.join(self.mask_dir,f) for f in os.listdir(self.mask_dir)]
        self.img_transform = image_transform
        self.mask_transform = mask_transform
        # #确保mask和img对应
        # self.img_path.sort()
        # self.mask_path.sort()

    def __len__(self):
        return len(self.img_path) * self.n_vol_size

    def __getitem__(self, item):
        file_idx = item // self.n_vol_size
        img_idx = np.mod(item,self.n_vol_size)
        file_path = self.img_path[file_idx]
        file_t1 = self.find_file_with_str(file_path,"_t1.")
        if len(file_t1) != 1:
            raise ValueError(f"{file_path}中没有t1文件或者t1文件数大于1")
        file_seg = self.find_file_with_str(file_path, "_seg.")
        if len(file_seg) != 1:
            raise ValueError(f"{file_path}中没有mask文件或者mask文件数大于1")
        file_t1ce = self.find_file_with_str(file_path, "_t1ce.")
        if len(file_t1ce) != 1:
            raise ValueError(f"{file_path}中没有_t1ce文件或者_t1ce文件数大于1")
        file_t2 = self.find_file_with_str(file_path, "_t2.")
        if len(file_t2) != 1:
            raise ValueError(f"{file_path}中没有_t2文件或者_t2文件数大于1")
        file_flair = self.find_file_with_str(file_path, "_flair.")
        if len(file_flair) != 1:
            raise ValueError(f"{file_path}中没有_flair文件或者_flair文件数大于1")

        img_t1 = sitk.ReadImage(os.path.join(file_path,file_t1[0]))
        # sitk.Show(img_t1,title="img")
        img_all = sitk.GetArrayFromImage(img_t1)
        t1 = img_all[img_idx:img_idx+1,:,:].astype(np.float32)

        img_t2 = sitk.ReadImage(os.path.join(file_path, file_t2[0]))
        # sitk.Show(img_t1,title="img")
        img_all = sitk.GetArrayFromImage(img_t2)
        t2 = img_all[img_idx:img_idx+1, :, :].astype(np.float32)

        img_t1ce = sitk.ReadImage(os.path.join(file_path, file_t1ce[0]))
        # sitk.Show(img_t1,title="img")
        img_all = sitk.GetArrayFromImage(img_t1ce)
        t1ce = img_all[img_idx:img_idx+1, :, :].astype(np.float32)

        img_flair = sitk.ReadImage(os.path.join(file_path, file_flair[0]))
        # sitk.Show(img_t1,title="img")
        img_all = sitk.GetArrayFromImage(img_flair)
        flair = img_all[img_idx:img_idx+1, :, :].astype(np.float32)

        img = np.concatenate([t1,t2,t1ce,flair],axis=0)

        # img = torch.cat([img_t1,img_t2,img_t1ce,img_flair],dim=0)



        msk_seg = sitk.ReadImage(os.path.join(file_path,file_seg[0]))
        mask_all = sitk.GetArrayFromImage(msk_seg)
        mask = mask_all[img_idx,:,:].astype(np.float32)
        # torch.randint(0,)
        # img = Image.open(self.img_path[item]).convert('RGB')
        # mask = Image.open(self.mask_path[item]).convert('L')
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        # img = torch.unsqueeze(torch.tensor(img),0)
        # mask = torch.unsqueeze(torch.tensor(mask),0)
        mask = torch.tensor(mask,dtype=torch.long)
        return img,mask

    def find_file_with_str(self,path,str):
        files = os.listdir(path)
        file_list = []
        for file in files:
            if os.path.isfile(os.path.join(path,file)) and str in file:
                file_list.append(file)
        return file_list