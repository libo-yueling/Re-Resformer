from typing import List, Optional, Union
from mmengine import fileio
from mmengine.logging import MMLogger
from mmpretrain.registry import DATASETS
from .custom import CustomDataset
from .categories import MAYTXT_CATEGORIES

@DATASETS.register_module()
class Maytxt(CustomDataset):
    """
    The dataset supports two kinds of directory format,
    ...
    """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': MAYTXT_CATEGORIES}

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
                    'Since the ImageNet1k test set does not provide label '
                    'annotations, `with_label` is set to False')
                kwargs['with_label'] = False

            data_prefix = split if data_prefix == '' else data_prefix

            if ann_file == '':
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = _ann_path  # 使用正确的注释文件路径

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
