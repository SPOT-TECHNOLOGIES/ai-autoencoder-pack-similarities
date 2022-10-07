'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''

from torch.utils.data import Dataset

## Dataset class inheritance to read collections of images from folder 
class customDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = sorted(os.listdir(main_dir))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image


## Dataset class inheritance to read collections of jsons 
class jsonDataset(Dataset):
    def __init__(self, main_dir,cfg, transform=None):
        self.main_dir = main_dir
        self.cfg = cfg
        self.transform = transform
        self.all_jsons = sorted(os.listdir(main_dir))
        self.index = np.arange(len(self.all_jsons))

    def __len__(self):
        return len(self.all_jsons)

    def __getitem__(self, idx):
        json_loc = os.path.join(self.main_dir, self.all_jsons[idx])
        json_data = json.load(open(json_loc))
        moments = json_data["moments"]
        person_id = json_data["person_id"]
        coverage = json_data["coverage"]
        index = self.index[idx]

        img_moments = []
        for m in moments:
            _,img = read_base64(m,self.cfg)
            img_tensor =  torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            img_moments.append(img_tensor)
        
        # if self.transform is not None:
        #     tensor_image = self.transform(image)

        return img_moments, person_id, coverage,index