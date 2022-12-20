'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''

from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import base64
import cv2
from config import *
import torch

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
        self.transform = transform
        collection_id =[]
        collection = []
        for coll in json_file["data"]["collection"]:
            collection_id.append(coll["pallet_id"]) 
            collection.append(coll["base64"])
        self.collection_id = collection_id
        self.collection = collection
        self.ncollect = len(collection)

    def __len__(self):
        return self.ncollect

    def __getitem__(self, idx):
        img_id = self.collection_id[idx]
        img_b64 = self.collection[idx]

        img_b64 = base64.b64decode(img_b64)
        im_arr = np.frombuffer(img_b64,dtype = np.uint8)
        img_dec = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)[:, :, ::-1]
        img_dec_res = cv2.resize(img_dec, tuple([IMG_WIDTH,IMG_HEIGHT]), interpolation=cv2.INTER_CUBIC)
        img_tensor =  torch.as_tensor(img_dec_res.astype("float32").transpose(2, 0, 1))
        
        
        if self.transform is not None:
            tensor_image = self.transform(image)

        return img_tensor, img_id

## Dataset class inheritance to read collections of jsons 
class customDataset(Dataset):
    def __init__(self, collection, transform=None):
        self.transform = transform
        collection_id =[]
        imgs = []
        for coll in collection:
            collection_id.append(coll["pallet_id"]) 
            imgs.append(coll["base64"])
        self.collection_id = collection_id
        self.imgs = imgs
        self.ncollect = len(imgs)
        self.labels = np.arange(self.ncollect)

    def __len__(self):
        return self.ncollect

    def __getitem__(self, idx):
        img_id = self.collection_id[idx]
        img_b64 = self.imgs[idx]
        label = self.labels[idx]

        img_b64 = base64.b64decode(img_b64)
        im_arr = np.frombuffer(img_b64,dtype = np.uint8)
        img_dec = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)[:, :, ::-1]
        img_dec_res = cv2.resize(img_dec, tuple([IMG_WIDTH,IMG_HEIGHT]), interpolation=cv2.INTER_CUBIC)
        img_tensor =  torch.as_tensor(img_dec_res.astype("float32").transpose(2, 0, 1))
        
        return img_tensor, img_id, label 