'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''

import torchvision.transforms as T
from config import IMG_WIDTH, IMG_HEIGHT

transforms_val = T.Compose([T.Resize((IMG_WIDTH,IMG_HEIGHT)),T.ToTensor()])
transforms_train = T.Compose([T.Resize((IMG_WIDTH,IMG_HEIGHT)),T.AutoAugment(T.AutoAugmentPolicy.SVHN),T.ToTensor()])