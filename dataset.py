from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class LandscapePaintDataset(Dataset):
    def __init__(self, landscape_dir, paint_dir, transforms=None):
        self.landscape_dir = landscape_dir
        self.paint_dir = paint_dir
        self.transform = transforms

        self.landscape_images = os.listdir(landscape_dir)
        self.paint_images = os.listdir(paint_dir)

        self.length_dataset = max(len(self.landscape_images), len(self.paint_images))

        self.landscape_len = len(self.landscape_images)
        self.paint_len = len(self.paint_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):

        landscape_image = self.landscape_images[index % self.landscape_len]
        paint_image = self.paint_images[index % self.paint_len]

        landscape_path = os.path.join(self.landscape_dir, landscape_image)
        paint_path = os.path.join(self.paint_dir, paint_image)

        landscape_image = np.array(Image.open(landscape_path).convert('RGB'))
        paint_image = np.array(Image.open(paint_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=landscape_image,
                                           image0=paint_image)

            landscape_image = augmentations['image']
            paint_image = augmentations['image0']

            return landscape_image, paint_image