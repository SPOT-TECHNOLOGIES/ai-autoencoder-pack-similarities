"""
@author:  wramirez
@contact: william.ramirez@spotcloud.io
"""

import io
import json

from flask import Flask, jsonify, request

import sys
import os 
from termcolor import colored
import base64
import PIL
import numpy as np
import cv2

import torch
from autoencoder.utils import *

app = Flask(__name__)

@app.route("/package_sim",methods=['POST'])

# http://ec2-52-22-86-23.compute-1.amazonaws.com:443/compare

def compare():

	#reading images
	json_request = request.get_json()

	img_req2 = json_request["pallet_a"]
	img_req1 = json_request["pallet_b"]

	img_bytes1 = base64.b64decode(img_req1)
	img_bytes2 = base64.b64decode(img_req2)

	im_arr1 = np.frombuffer(img_bytes1,dtype = np.uint8)
	im_arr2 = np.frombuffer(img_bytes2,dtype = np.uint8)

	img_dec1 = cv2.imdecode(im_arr1, \
		flags=cv2.IMREAD_COLOR)[:, :, ::-1]
	img_dec2 = cv2.imdecode(im_arr2, \
		flags=cv2.IMREAD_COLOR)[:, :, ::-1]

	img_dec_res1 = cv2.resize(img_dec1, \
		tuple([IMG_WIDTH,IMG_HEIGHT]), \
		interpolation=cv2.INTER_CUBIC)
	img_dec_res2 = cv2.resize(img_dec2, \
		tuple([IMG_WIDTH,IMG_HEIGHT]), \
		interpolation=cv2.INTER_CUBIC)

	img_t1 =  torch.as_tensor(img_dec_res1.astype("float32") \
		.transpose(2, 0, 1))
	img_t2 =  torch.as_tensor(img_dec_res2.astype("float32") \
		.transpose(2, 0, 1))

	encoder = create_encoder("cpu")
	f1 = extract_image_feature(img_t1,encoder,"cpu")
	f2 = extract_image_feature(img_t2,encoder,"cpu")

	sim = np.matmul(f1,f2.transpose())[0][0]

	return str(sim)


if __name__== "__main__":
	# app.run(debug= False, port= 6006)
	# app.run(debug= False, port= 443,host='0.0.0.0')
	application.run(debug= True, host='0.0.0.0', port=8082)



