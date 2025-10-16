import os
import torch
from typing import List, Optional, Union
from PIL import Image
from mmengine import get_file_backend
from mmpretrain.registry import DATASETS
from .custom import CustomDataset
from torchvision import transforms
from mmpretrain.structures.data_sample import DataSample
import torchvision.transforms.functional as TF

@DATASETS.register_module()
class Reg(CustomDataset):
    """The Rac Dataset for image roughness detection.

    Args:
        data_root (str): The root directory for Rac dataset.
        split (str, optional): The dataset split (e.g., "train", "val"). Default to "".
        data_prefix (Union[str, dict], optional): Data prefix for images. Default to "".
        ann_file (str, optional): Annotation file path. Default to "".
        metainfo (Optional[dict], optional): Metadata for the dataset. Default to None.
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        self.split = split
        self.data_prefix = data_prefix if data_prefix else split
        self.backend = get_file_backend(data_root, enable_singleton=True)

        # Validate the split and load data info
        if split:
            self.data_info = self.load_data_info(data_root, split)

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
        ])

        super().__init__(data_root=data_root, data_prefix=data_prefix, ann_file=ann_file, metainfo=metainfo, **kwargs)

    def load_data_info(self, data_root: str, split: str):
        """Load image paths and associated metadata from the specified split."""
        data_info = []
        ann_file_path = os.path.join(data_root, f'{split}.txt')

        # Check if annotation file exists
        if not os.path.isfile(ann_file_path):
            raise FileNotFoundError(f"Annotation file {ann_file_path} does not exist.")

        with open(ann_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    print(f"Invalid line in annotation file: {line}")
                    continue

                img_path = parts[0]
                try:
                    roughness_value = float(parts[1])
                    contrast_value = float(parts[2])
                except ValueError as e:
                    print(f"Value error in line: {line}, error: {e}")
                    continue

                # Check if image path exists
                full_img_path = os.path.join(data_root, img_path)
                if not os.path.isfile(full_img_path):
                    print(f"Image file {full_img_path} does not exist.")
                    continue

                # 存储格式：(img_path, roughness_value, contrast_value)
                data_info.append((img_path, roughness_value, contrast_value))

        return data_info

    def load_image(self, img_path):
        """Load the image from the given path and apply transformations."""
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        image = image.resize((224, 224), Image.LANCZOS)
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        """Get item by index."""
        if index >= len(self.data_info) or index < 0:
            raise IndexError(f"Index {index} out of range. Data info length: {len(self.data_info)}")

        img_path, roughness_value, correlation_value = self.data_info[index]
        full_img_path = os.path.join(self.data_root, img_path)
        image = self.load_image(full_img_path)

        if image is None:
            raise RuntimeError(f"Failed to load image at index {index}: {img_path}")

        # Create DataSample object and set properties
        gray_image = TF.rgb_to_grayscale(image)  # 转换为单通道灰度图
        data_sample = DataSample().set_gt_roughness(roughness_value)
        data_sample = data_sample.set_gt_correlation(correlation_value)
        data_sample.img_shape = image.shape  # Set image shape
        data_sample.sample_idx = index  # Set sample index

        # Return a dictionary containing 'inputs' and 'data_samples'
        return {
            'inputs': image,  # Image tensor for processing
            'data_samples': data_sample  # Additional sample information
        }

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data_info)
