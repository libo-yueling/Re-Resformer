from typing import List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .custom import CustomDataset
from .categories import PSB_CATEGORIES

@DATASETS.register_module()
class Psb(CustomDataset):
    """
        The dataset supports two kinds of directory format,
        ::
            imagenet
            ├── train
            │   ├──class_x
            |   |   ├── x1.jpg
            |   |   ├── x2.jpg
            |   |   └── ...
            │   ├── class_y
            |   |   ├── y1.jpg
            |   |   ├── y2.jpg
            |   |   └── ...
            |   └── ...
            ├── val
            │   ├──class_x
            |   |   └── ...
            │   ├── class_y
            |   |   └── ...
            |   └── ...
            └── test
                ├── test1.jpg
                ├── test2.jpg
                └── ...

        or ::

            imagenet
            ├── train
            │   ├── x1.jpg
            │   ├── y1.jpg
            │   └── ...
            ├── val
            │   ├── x3.jpg
            │   ├── y3.jpg
            │   └── ...
            ├── test
            │   ├── test1.jpg
            │   ├── test2.jpg
            │   └── ...
            └── meta
                ├── train.txt
                └── val.txt


        Args:
            data_root (str): The root directory for ``data_prefix`` and
                ``ann_file``. Defaults to ''.
            split (str): The dataset split, supports "train", "val" and "test".
                Default to ''.
            data_prefix (str | dict): Prefix for training data. Defaults to ''.
            ann_file (str): Annotation file path. Defaults to ''.
            metainfo (dict, optional): Meta information for dataset, such as class
                information. Defaults to None.
            **kwargs: Other keyword arguments in :class:`CustomDataset` and
                :class:`BaseDataset`.


        Examples:
            >>> from mmpretrain.datasets import ImageNet
            >>> train_dataset = Psb(data_root='D:/classiyf-module/mmpretrain-main/dataset/45g', split='train')
            >>> train_dataset
            Dataset ImageNet
                Number of samples:  1281167
                Number of categories:       3
                Root of dataset:    data/imagenet
            >>> test_dataset = Psb(data_root='D:/classiyf-module/mmpretrain-main/dataset/45g', split='test')
            >>> test_dataset

            >>> val_dataset = Psb(data_root='D:/classiyf-module/mmpretrain-main/dataset/45g', split='val')
            >>> val_dataset
            Dataset ImageNet
                Number of samples:  50000
                Number of categories:       3
                Root of dataset:    data/imagenet
        """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': PSB_CATEGORIES}

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}

        if split:
            splits = ['train', 'val', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"

            if split == 'test':
                logger = MMLogger.get_current_instance()
                logger.info(
                    'Since the ImageNet1k test set does not provide label'
                    'annotations, `with_label` is set to False')
                kwargs['with_label'] = False

            data_prefix = split if data_prefix == '' else data_prefix

            if ann_file == '':
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
