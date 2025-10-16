import os
from typing import List, Optional, Union
from PIL import Image
from mmengine import get_file_backend
from mmpretrain.registry import DATASETS
from .custom import CustomDataset
from torchvision import transforms
from mmpretrain.structures.data_sample import DataSample  # 确保导入 DataSample 类


@DATASETS.register_module()
class Rac(CustomDataset):
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
            transforms.ToTensor(),  # Convert to tensor
        ])

        super().__init__(data_root=data_root, data_prefix=data_prefix, ann_file=ann_file, metainfo=metainfo, **kwargs)

    def load_data_info(self, data_root: str, split: str):
        """Load image paths and associated metadata from the specified split."""
        data_info = []
        ann_file_path = os.path.join(data_root, f'{split}.txt')

        # Check if annotation file exists
        if not os.path.isfile(ann_file_path):
            raise FileNotFoundError(f"Annotation file {ann_file_path} does not exist.")

        # print(f"Loading data from {ann_file_path}...")
        with open(ann_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    print(f"Invalid line in annotation file: {line}")
                    continue

                img_path = parts[0]
                class_label = int(parts[1])  # Classification label
                roughness_value = float(parts[2])  # Roughness value

                # Check if image path exists
                full_img_path = os.path.join(data_root, img_path)
                if not os.path.isfile(full_img_path):
                    print(f"Image file {full_img_path} does not exist.")
                    continue

                data_info.append((img_path, class_label, roughness_value))

        return data_info

    def load_image(self, img_path):
        """Load the image from the given path and apply transformations."""
        try:
            image = Image.open(img_path).convert('RGB')
            # print(f"Successfully loaded image {img_path}.")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None  # 或者抛出异常，具体取决于你的设计

        image = image.resize((224, 224), Image.LANCZOS)
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        """Get item by index."""
        if index >= len(self.data_info) or index < 0:
            raise IndexError(f"Index {index} out of range. Data info length: {len(self.data_info)}")

        img_path, class_label, roughness_value = self.data_info[index]
        image = self.load_image(img_path)

        if image is None:
            raise RuntimeError(f"Failed to load image at index {index}: {img_path}")

        # 创建 DataSample 对象并设置属性
        data_sample = DataSample().set_gt_label(class_label).set_gt_roughness(roughness_value)
        data_sample.img_shape = image.shape  # 设置图像形状
        data_sample.sample_idx = index  # 设置样本索引

        # 返回一个字典，确保包含 'inputs' 键，并满足后续处理需要的格式
        return {
            'inputs': image,  # 包含图像张量，供 backbone 处理
            'data_samples': data_sample  # 包含样本的附加信息
        }


    def __len__(self):
        """Return the total number of samples."""
        return len(self.data_info)
