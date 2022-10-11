import os 

os.sys.path.append('/media/william/HDD_WILL/Documents/SPOT/ai-package-similarities-pt/autoencoder/')

from engine import *
from data import *

rain_loader, val_loader, test_loader = build_data()