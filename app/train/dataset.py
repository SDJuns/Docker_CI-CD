from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomOrderImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, custom_order=None):
        self.custom_order = custom_order
        super().__init__(root, transform=transform, target_transform=target_transform)

    def find_classes(self, directory):
        if self.custom_order is not None:
            classes = self.custom_order
        else:
            classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
            classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

class CustomSubset(Dataset):
    def __init__(self, image_folder, indices, transform):
        self.image_folder = image_folder
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        path, label = self.image_folder.samples[actual_idx]
        image = self.image_folder.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label
