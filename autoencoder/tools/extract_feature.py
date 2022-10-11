import os 

os.sys.path.append('../ai-package-similarities-pt/autoencoder/')

from engine import *
from modeling import *
import torch
from config import *

def create_encoder(device):
	encoder = convEncoder()
	encoder.load_state_dict(torch.load(ENC_MODEL_PATH))
	encoder.to(device)
	encoder.eval()

	return encoder

