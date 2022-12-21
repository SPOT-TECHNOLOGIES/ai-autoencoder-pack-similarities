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
from autoencoder.data.custom_dataset import customDataset
from autoencoder.config import *

app = Flask(__name__)

@app.route("/similarity",methods=['POST'])


def compare():

        json_request = request.get_json()

        collection = json_request["data"]["CD"]
        queries = json_request["data"]["Local"]

        collection_dataset = customDataset(collection)
        query_dataset = customDataset(queries)

        print("coll size",collection_dataset.ncollect)
        
        collection_loader = torch.utils.data.DataLoader(
                collection_dataset,batch_size=collection_dataset.ncollect)
        
        query_loader = torch.utils.data.DataLoader(
                query_dataset,batch_size=query_dataset.ncollect)

        if torch.cuda.is_available():
                device = "cuda"
                print("GPU AVAILABLE")
        else:
                device = "cpu"
                print("CPU AVAILABLE")

        encoder = create_encoder(device)

        collection_emb, pallet_ids, coll_labels = create_embedding_v2(encoder, collection_loader,EMBEDDING_DIM, device)
        collection_emb = collection_emb.reshape((collection_emb.shape[0],-1))


        query_emb, pallet_ids, query_labels = create_embedding_v2(encoder, query_loader,EMBEDDING_DIM, device)
        query_emb = query_emb.reshape((query_emb.shape[0],-1))

        n_neighb = query_emb.shape[0]

        knn = NearestNeighbors(n_neighbors=n_neighb, metric="cosine")
        knn.fit(collection_emb)

        dist, indices = knn.kneighbors(query_emb)
        indices_list = indices.tolist()
        sim = 1.0 - dist

        # print("coll_labels: ",coll_labels)
        # print("query_labels: ",query_labels)
        # print("indices: ",indices)
        print("sim: ",sim)

        # print("max: ")
        
        M = []
        for i,ql in enumerate(query_labels):
                for j in range(sim.shape[1]):
                        M.append([ql,indices[i,j],sim[i,j]])

        qlabels = []
        idx = []
        simx = []
        cont = 0
        
        
        print("M: ",M)

        for i in range(query_labels.size):
                max = 0
                for j in range(len(M)):
                        if M[j][0] not in qlabels and M[j][1] not in idx:
                                # print(M[j])
                                if  M[j][2] >= max:
                                        max = M[j][2]
                                        maxq = M[j][0]
                                        maxi = M[j][1]
                qlabels.append(maxq)
                idx.append(maxi)   
                simx.append(max)

        sortind = np.argsort(qlabels)

        print(np.array(qlabels)[sortind])
        print(np.array(idx)[sortind])
        print(np.array(simx)[sortind])

        res = jsonify({"data":{"indexes":np.array(idx)[sortind].tolist(),
            "similarities":np.array(simx)[sortind].tolist()}})

        return res


if __name__== "__main__":
#       app.run(debug= False, port= 6006)
        # app.run(debug= False, port= 443,host='0.0.0.0')
        app.run(debug= True,host='0.0.0.0', port=8082)



