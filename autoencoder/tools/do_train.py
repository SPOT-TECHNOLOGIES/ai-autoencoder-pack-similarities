import os 

os.sys.path.append('/media/william/HDD_WILL/Documents/SPOT/ai-package-similarities-pt/autoencoder/')

from engine import *
from data import *

train_loader, val_loader, test_loader = build_data()
train(train_loader,val_loader,"cpu")