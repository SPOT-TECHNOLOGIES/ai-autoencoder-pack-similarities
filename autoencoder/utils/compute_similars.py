'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''


from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import torch

def compute_similar_images(image_tensor, num_images, embedding,encoder, device):
    """
    Given an image and number of similar images to search.
    Returns the num_images closest neares images.
    Args:
    image: Image whose similar images are to be found.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
        
    flattened_embedding = image_embedding.reshape((1, -1))

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()

    return indices_list,flattened_embedding