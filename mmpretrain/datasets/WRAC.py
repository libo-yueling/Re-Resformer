import os
import numpy as np
from PIL import Image
from typing import List, Optional, Union
from mmengine import get_file_backend
from mmpretrain.registry import DATASETS
from .custom import CustomDataset
from torchvision import transforms
from mmpretrain.structures.data_sample import DataSample
import pywt  # 需要使用pywt库来进行小波变换


# 小波变换类的实现
class ApplyWaveletTransform:
    def __init__(self, wavelet='haar', mode='symmetric'):
        self.wavelet = wavelet
        self.mode = mode

    def __call__(self, image):
        """Apply the wavelet transform to the image."""
        # Convert PIL Image to numpy array
        image = np.array(image)

        # 进行小波变换（这里我们将图像分解为低频和高频成分）
        coeffs2 = pywt.dwt2(image, self.wavelet, mode=self.mode)
        LL, (LH, HL, HH) = coeffs2

        # 你可以选择如何处理分解后的小波系数
        # 例如，这里返回的是低频部分（LL），但你也可以选择其他方式来组合这些系数
        return np.array(LL, dtype=np.float32)


@DATASETS.register_module()
class Wrac(CustomDataset):
    """The Wrac Dataset for image classification and wavelet transform."""

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

        # Define transformations with ApplyWaveletTransform
        self.transform = transforms.Compose([
            ApplyWaveletTransform(),  # 使用自定义的小波变换
            transforms.ToTensor(),    # Convert to tensor
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
                if len(parts) < 2:
                    print(f"Invalid line in annotation file: {line}")
                    continue

                img_path = parts[0]
                class_label = int(parts[1])  # Classification label

                # Check if image path exists
                full_img_path = os.path.join(data_root, img_path)
                if not os.path.isfile(full_img_path):
                    print(f"Image file {full_img_path} does not exist.")
                    continue

                data_info.append((img_path, class_label))

        return data_info

    def load_image(self, img_path):
        """Load the image from the given path and apply transformations."""
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        image = image.resize((224, 224), Image.LANCZOS)

        # Apply transformations (ApplyWaveletTransform + ToTensor)
        image_transformed = self.transform(image)
        return image_transformed

    def __getitem__(self, index):
        """Get item by index."""
        if index >= len(self.data_info) or index < 0:
            raise IndexError(f"Index {index} out of range. Data info length: {len(self.data_info)}")

        img_path, class_label = self.data_info[index]
        image = self.load_image(img_path)

        if image is None:
            raise RuntimeError(f"Failed to load image at index {index}: {img_path}")

        # Create DataSample object and set classification label
        data_sample = DataSample().set_gt_label(class_label)
        data_sample.img_shape = image.shape  # Set image shape
        data_sample.sample_idx = index  # Set sample index

        # Return a dictionary that includes 'inputs' (image tensor) and 'data_samples' (meta info)
        return {
            'inputs': image,  # Image tensor for the model's backbone
            'data_samples': data_sample  # Data sample with additional meta info
        }

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data_info)
