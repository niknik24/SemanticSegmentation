from collections import defaultdict
import copy
import random
import numpy as np
import os
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as T

from MapillaryDataset import MapillaryDataset, DinamicObjInferenceDataset
import utils
import visualize


def train():
    import warnings
    warnings.filterwarnings("ignore")


    images_directory_for_dataset = os.path.join("training/images")
    masks_directory_for_dataset = os.path.join("training/labels")
    val_directory_im = os.path.join("validation/images")
    val_directory_mask = os.path.join("validation/labels")

    images_filenames = list(sorted(os.listdir(images_directory_for_dataset)))
    val_filenames = list(sorted(os.listdir(val_directory_im)))


    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_dataset = MapillaryDataset(images_filenames, images_directory_for_dataset, masks_directory_for_dataset,
                                            transform=train_transform)


    val_transform = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

    val_dataset = MapillaryDataset(val_filenames, val_directory_im, val_directory_mask,
                                   transform=val_transform)

    visualize.display_image_grid(images_filenames[:4], images_directory_for_dataset,
                                    masks_directory_for_dataset)

    random.seed(42)
    visualize.visualize_augmentations(train_dataset, idx=5)

    params = {
        "model": "UNet16",
        "num_classes": 2,
        "device": "cuda",
        "lr": 2e-5,
        "batch_size": 1,
        "num_workers": 4,
        "epochs": 10,
        "amp": True,
    }

    import ternausnet.models

    def create_model(params):
        model = getattr(ternausnet.models, params["model"])(pretrained=True)
        model = model.to(params["device"])
        return model

    #model = getattr(ternausnet.models, 'UNet16')(pretrained=True)
    #model = model.to(params["device"], non_blocking=True)
    model = create_model(params)
    #model.load_state_dict(torch.load("saved_models/final.pth"))
    model = utils.train_and_validate(model, train_dataset, val_dataset, params)


    test_directory = os.path.join("testing/images")
    test_images_filenames = list(sorted(os.listdir(test_directory)))

    test_transform = A.Compose(
        [#A.LongestMaxSize(512),
         A.Resize(512, 512),
         #A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ToTensorV2(), ]
    )
    test_dataset = DinamicObjInferenceDataset(test_images_filenames, test_directory,
                                              transform=test_transform, )
    predictions = utils.predict(model, params, test_dataset, batch_size=1)


    predicted_masks = []
    for predicted_256x256_mask, original_height, original_width in predictions:
        full_sized_mask = cv2.resize(predicted_256x256_mask, (original_height, original_width),
                                     interpolation=cv2.INTER_NEAREST
                                     )
        predicted_masks.append(full_sized_mask)

    #if you have masks for test use normal image_grid
    visualize.display_image_grid_for_pred(test_images_filenames, test_directory,
                                 predicted_masks=predicted_masks)

