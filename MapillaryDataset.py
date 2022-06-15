import os
from torch.utils.data import Dataset
import cv2
from matplotlib import pyplot as plt
from os.path import splitext
from os import listdir
import numpy as np
from torch.utils import data
from utils import preprocess_mask
from PIL import Image

class MapillaryDataset(data.Dataset):
    def __init__(self,  images_filenames, img_dir, mask_dir, transform=None):
        self.images_filenames=images_filenames
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ids = []
        self.ids = [splitext(file)[0] for file in listdir(img_dir) if not file.startswith('.')]
        self.transform = transform

        print(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, is_mask):
        img_ndarray = np.array(pil_img)
        if is_mask:
            img_ndarray = np.where((img_ndarray != 19) & (img_ndarray != 20) & (img_ndarray != 55) & (img_ndarray != 54)
                                   & (img_ndarray != 57) & (img_ndarray != 61) & (img_ndarray != 59)
                                  & (img_ndarray != 52) & (img_ndarray != 58), 0, img_ndarray)
            img_ndarray = np.where((img_ndarray == 19) | (img_ndarray == 20), 1, img_ndarray)
            img_ndarray = np.where((img_ndarray == 55) | (img_ndarray == 54)
                                   | (img_ndarray == 57) | (img_ndarray == 61) | (img_ndarray == 59)
                                   | (img_ndarray == 52) | (img_ndarray == 58), 2, img_ndarray)
            img_ndarray=img_ndarray/255
        if not is_mask:
            img_ndarray = img_ndarray

        return img_ndarray


    def __getitem__(self, i):
        image_filename = self.images_filenames[i]
        image = cv2.imread(os.path.join(self.img_dir, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            os.path.join(self.mask_dir, image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED,
        )

        img1 = self.preprocess(image, False)
        mask = preprocess_mask(mask)
        plt.imshow(mask)
        plt.show()
        if self.transform is not None:
            transformed = self.transform(image=img1, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

            return image, mask

        return img1, mask

class DinamicObjInferenceDataset(Dataset):
    def __init__(self, images_filenames, images_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image1 = cv2.imread(os.path.join(self.images_directory, image_filename))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image = self.images_directory + "/" + self.images_filenames[idx]
        image = Image.open(image)
        original_size = tuple(image.size)
        if self.transform is not None:
            transformed = self.transform(image=image1)
            image = transformed["image"]
        return image, original_size
