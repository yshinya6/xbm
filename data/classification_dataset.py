import os

import torch
import torchvision
from torch.utils.data import random_split


class GenericImageDataset(torchvision.datasets.ImageFolder):

    def __init__(self, dataset_name, transform, root, test=False, test_split="test"):
        self.dataset_name = dataset_name
        if test:
            split_data_dir = os.path.join(root, test_split)
        else:
            split_data_dir = os.path.join(root, "train")
        super(GenericImageDataset, self).__init__(root=split_data_dir, transform=transform)

    def get_dataset_name(self):
        return self.dataset_name


def setup_image_classification_dataset(
    dataset_name, transform_train, transform_test, root, train_val_split_ratio=0.9, test_split="test"
):
    seed = torch.seed()
    torch.manual_seed(42)  # Ensure fixed seed to randomly split datasets
    all_train_dataset = GenericImageDataset(dataset_name=dataset_name, transform=transform_train, root=root, test=False)
    test_dataset = GenericImageDataset(
        dataset_name=dataset_name, transform=transform_test, root=root, test=True, test_split=test_split
    )
    train_size = int(len(all_train_dataset) * train_val_split_ratio)
    val_size = len(all_train_dataset) - train_size
    train_dataset, val_dataset = random_split(all_train_dataset, [train_size, val_size])
    val_dataset.transform = test_dataset.transform
    torch.manual_seed(seed)
    return train_dataset, val_dataset, test_dataset
