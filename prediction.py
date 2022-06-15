import ternausnet.models
import os
import torch
from MapillaryDataset import MapillaryDataset, DinamicObjInferenceDataset
import utils
import visualize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def prediction(testdir, params):

    def create_model(params):
        model = getattr(ternausnet.models, params["model"])(pretrained=True)
        model = model.to(params["device"])
        return model

    model = getattr(ternausnet.models, 'UNet16')(pretrained=True)
    model = model.to(params["device"], non_blocking=True)
    model.load_state_dict(torch.load("saved_models/final.pth"))

    test_directory = os.path.join(testdir)
   #test_dir_mask = os.path.join(testdir+"labels)
    test_images_filenames = list(sorted(os.listdir(test_directory)))

    test_transform = A.Compose(
        [  # A.LongestMaxSize(512),
            A.Resize(512, 512),
            # A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
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

    # if you have masks for test use normal image_grid
    visualize.display_image_grid_for_pred(test_images_filenames, test_directory,
                                          predicted_masks=predicted_masks)