import os

import torch
import torchvision
from torch.utils.data import random_split
from PIL import Image
import pdb


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class GenericImageDataset(torchvision.datasets.ImageFolder):

    def __init__(self, dataset_name, data_args, transform, root, test=False, test_split="test"):
        self.dataset_name = dataset_name
        self.data_args = data_args
        if test:
            split_data_dir = os.path.join(root, test_split)
        else:
            split_data_dir = os.path.join(root, "train")
        super(GenericImageDataset, self).__init__(root=split_data_dir, transform=transform)

    def get_dataset_name(self):
        return self.dataset_name

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)  # PIL Image (RGB)
        # if self.transform is not None:
        #     image = self.transform(image)  # PIL
        processor = self.data_args.image_processor
        if self.data_args.image_aspect_ratio == "pad":
            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def setup_image_classification_dataset(
    dataset_name, data_args, transform_train, transform_test, root, train_val_split_ratio=0.9, test_split="test"
):
    seed = torch.seed()
    torch.manual_seed(42)  # Ensure fixed seed to randomly split datasets
    all_train_dataset = GenericImageDataset(
        dataset_name=dataset_name, data_args=data_args, transform=transform_train, root=root, test=False
    )
    test_dataset = GenericImageDataset(
        dataset_name=dataset_name,
        data_args=data_args,
        transform=transform_test,
        root=root,
        test=True,
        test_split=test_split,
    )
    train_size = int(len(all_train_dataset) * train_val_split_ratio)
    val_size = len(all_train_dataset) - train_size
    train_dataset, val_dataset = random_split(all_train_dataset, [train_size, val_size])
    val_dataset.transform = test_dataset.transform
    torch.manual_seed(seed)
    return train_dataset, val_dataset, test_dataset
