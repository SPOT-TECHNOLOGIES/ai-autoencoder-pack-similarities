'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''
import torch
import numpy as np
import torch.nn.functional as F

def create_embedding(encoder, full_loader, embedding_dim, device):
    encoder.eval()
    embedding = torch.randn(embedding_dim)
    indexes = []

    with torch.no_grad():
        for batch_idx, (img, img_id) in enumerate(full_loader):
            img = img.to(device)
            enc_output = encoder(img).cpu()
            embedding = torch.cat((embedding, enc_output), 0)
            indexes.append(img_id.tolist())
            
    return embedding.cpu().detach().numpy()[1::,:],np.array(indexes).flatten()


def create_embedding_v2(encoder, full_loader, embedding_dim, device):
    encoder.eval()
    embedding = torch.randn(embedding_dim)
    indexes = []
    labels = []

    with torch.no_grad():
        for batch_idx, (img, img_id,label) in enumerate(full_loader):
            img = img.to(device)
            enc_output = encoder(img).cpu()
            enc_output_n = F.normalize(enc_output)
            embedding = torch.cat((embedding, enc_output_n), 0)
            indexes.append(img_id.tolist())
            labels.append(label.tolist())
            
    return embedding.cpu().detach().numpy()[1::,:],np.array(indexes).flatten(),np.array(labels).flatten()