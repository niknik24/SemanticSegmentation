import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import train
import prediction

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    params = {
        "device": "cuda",
        "num_workers": 4,
    }
    #prediction.prediction("testing/image", params)
