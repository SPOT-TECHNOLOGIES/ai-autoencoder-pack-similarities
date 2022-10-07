'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''


def create_embedding(encoder, full_loader, embedding_dim, device):
    encoder.eval()
    embedding = torch.randn(embedding_dim)

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(full_loader):
            train_img = train_img.to(device)
            enc_output = encoder(train_img).cpu()
            embedding = torch.cat((embedding, enc_output), 0)
    
    return embedding