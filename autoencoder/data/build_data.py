'''
Created by: William Ramírez
Email: william.ramirez@spotcloud.io

'''

from PIL import Image

from transforms import transforms_train, transforms_val
from .custom_dataset import customDataset
from config import IMG_TRAIN_PATH,IMG_VAL_PATH,IMG_TEST_PATH


def buil_data():

    print("------------ Creating Dataset ------------")
    train_dataset = customDataset(IMG_TRAIN_PATH, transforms_train)
    val_dataset = customDataset(IMG_VAL_PATH, transforms_val)
    test_dataset = customDataset(IMG_TEST_PATH, transforms_val)

    print("------------ Dataset Created ------------")
    print("------------ Creating DataLoader ------------")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE
    )

    return train_loader, val_loader, test_loader