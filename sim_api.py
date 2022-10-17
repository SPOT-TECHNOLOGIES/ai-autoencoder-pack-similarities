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
from autoencoder.data.custom_dataset import jsonDataset
from autoencoder.config import *

app = Flask(__name__)

@app.route("/similarity",methods=['POST'])


def compare():

	json_request = request.get_json()

	collection = jsonDataset(json_request)
	query_img_id = json_request["data"]["query"][0]["pallet_id"]
	query_img_b64 = json_request["data"]["query"][0]["base64"]

	img_bytes = base64.b64decode(query_img_b64)
	im_arr = np.frombuffer(img_bytes,dtype = np.uint8)
	img_dec = cv2.imdecode(im_arr, \
		flags=cv2.IMREAD_COLOR)[:, :, ::-1]
	img_dec_res = cv2.resize(img_dec, \
		tuple([IMG_WIDTH,IMG_HEIGHT]), \
		interpolation=cv2.INTER_CUBIC)
	query_tensor =  torch.as_tensor(img_dec_res.astype("float32") \
		.transpose(2, 0, 1))

	collection_loader = torch.utils.data.DataLoader(
		collection,batch_size=collection.ncollect)

	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"

	encoder = create_encoder(device)

	embedding, pallet_ids = create_embedding(encoder, collection_loader,EMBEDDING_DIM, device)
	embedding = embedding.reshape((embedding.shape[0],-1))

	k_idx,query_feat = compute_similar_images(query_tensor,2,\
		embedding,encoder,device)
	query_feat_n = query_feat/np.linalg.norm(query_feat)
	emb_n = np.array([embedding[i,:]/np.linalg.norm(embedding[i,:]) for i in k_idx[0]])
	sim = np.matmul(query_feat_n,emb_n.transpose())

	res = jsonify({"data":{"similars":[{"pallet_id":int(pallet_ids[k_idx[0][0]]) \
		,"similarity":float(sim[0][0])},{"pallet_id":int(pallet_ids[k_idx[0][1]]) \
		,"similarity":float(sim[0][1])}]}})

	return res


if __name__== "__main__":
	app.run(debug= False, port= 6006)
	# app.run(debug= False, port= 443,host='0.0.0.0')
	# app.run(debug= True,host='0.0.0.0', port=8082)



