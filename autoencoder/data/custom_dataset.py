'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''

from torch.utils.data import Dataset
import os
from PIL import Image

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
    def __init__(self, json_file, transform=None):
        self.cfg = cfg
        self.transform = transform
        collection_id =[]
        collection = []
        for coll in json_file["collection"]:
            collection_idx.append(coll["pallet_id"]) 
            collection.append(coll["base64"])
        self.collection_id = collection_id
        self.collection = collection

    def __len__(self):
        return len(self.all_jsons)

    def __getitem__(self, idx):
        img_id = self.collection_id[idx]
        img_b64 = self.collection[idx]

        _,img = read_base64(img_b64,self.cfg)
        img_tensor =  torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        
        
        if self.transform is not None:
            tensor_image = self.transform(image)

        return img_tensor, img_id