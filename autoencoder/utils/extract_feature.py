import os 

os.sys.path.append('../ai-autoencoder-pack-similarities/autoencoder/')

from engine import *
from modeling import *
import torch
from config import *
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

def create_encoder(device):
	encoder = convEncoder()
	encoder.load_state_dict(torch.load(ENC_MODEL_PATH +"/"+ MODEL_NAME ,map_location=torch.device(device)))
	encoder.to(device)
	encoder.eval()

	return encoder


def extract_image_feature(image,encoder,device):
	# image_tensor = T.Compose([T.Resize((IMG_WIDTH,IMG_HEIGHT)),T.ToTensor()])(image)
	# image_tensor = image_tensor.unsqueeze(0)
	image = image.to(device)

	with torch.no_grad():
		image_feature = encoder(image).cpu()\
		.detach().numpy()

	print("feature shape:",image_feature.shape)
	flat_image_feature = image_feature.reshape((1, -1))
	flat_image_feature = flat_image_feature/np.linalg.norm(flat_image_feature)
	return flat_image_feature

