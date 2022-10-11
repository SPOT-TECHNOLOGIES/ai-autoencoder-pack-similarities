'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''


from modeling import *
import torch
import torch.optim as optim
from config import *
from tqdm import tqdm

ENC_MODEL_PATH = "../outputs/weights/"
DEC_MODEL_PATH = "../outputs/weights/"

def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device="cuda"):
    
    encoder.train()
    decoder.train()

    for batch_idx, (train_img, target_img) in enumerate(train_loader):
        train_img = train_img.to(device)
        target_img = target_img.to(device)

        optimizer.zero_grad()

        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)

        loss = loss_fn(dec_output, target_img)
        loss.backward()

        optimizer.step()

    return loss.item()

def val_step(encoder, decoder, val_loader, loss_fn, device):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(val_loader):
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            loss = loss_fn(dec_output, target_img)

    return loss.item()


def train(train_loader,val_loader,device):

    loss_fn = nn.MSELoss()

    encoder = convEncoder()
    decoder = convDecoder()

    if torch.cuda.is_available():
        print("GPU Availaible moving models to GPU")
    else:
        print("Moving models to CPU")

    encoder.to(device)
    decoder.to(device)

    # print(device)

    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=LEARNING_RATE)

    # early_stopper = utils.EarlyStopping(patience=5, verbose=True, path=)
    max_loss = 9999

    print("------------ Training started ------------")

    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(
            encoder, decoder, train_loader, loss_fn, optimizer, device=device
        )
        print(f"Epochs = {epoch}, Training Loss : {train_loss}")
        val_loss = val_step(
            encoder, decoder, val_loader, loss_fn, device=device
        )

        # Simple Best Model saving
        if val_loss < max_loss:
            print("Validation Loss decreased, saving new best model")
            torch.save(encoder.state_dict(), ENC_MODEL_PATH)
            torch.save(decoder.state_dict(), DEC_MODEL_PATH)

        print(f"Epochs = {epoch}, Validation Loss : {val_loss}")

    print("Training Done")