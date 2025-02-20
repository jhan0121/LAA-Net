#-*- coding: utf-8 -*-
import os

import numpy as np
from glob import glob

from .builder import DATASETS
from .common import CommonDataset


@DATASETS.register_module()
class FF(CommonDataset):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def _load_from_path(self, split):
        print(self._cfg.DATA[self.split.upper()].ROOT)
        assert os.path.exists(self._cfg.DATA[self.split.upper()].ROOT), "Root path to dataset can not be None!"
        data = self._cfg["DATA"]
        data_type = data.TYPE
        fake_types = self._cfg.DATA[split.upper()]["FAKETYPE"]
        label_folders = self._cfg.DATA[split.upper()]["LABEL_FOLDER"]
        img_paths, labels, mask_paths, ot_props = [], [], [], []

        # Load image data for each type of fake techniques
        for idx, ft in enumerate(fake_types):
            if self.compression == 'c23':
                data_dir = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, ft, data_type, split)
                if not os.path.exists(data_dir):
                    raise ValueError("Data Directory can not be invalid!")
                
                for lb in label_folders:
                    if lb == "fake":
                        # Real 0, Fake 1
                        img_paths_ = glob(f'{data_dir}/{lb}/*.{self._cfg.IMAGE_SUFFIX}')
                    else:
                        #Only load real images once time
                        if idx != 0: continue
                        img_paths_ = glob(f'{data_dir}/{lb}/*.{self._cfg.IMAGE_SUFFIX}')
                    
                    img_paths.extend(img_paths_)
                    labels.extend(np.full(len(img_paths_), int(lb == 'fake')))
            elif self.compression == 'c0':
                data_dir = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, self.split, data_type, ft)
                if not os.path.exists(data_dir):
                    raise ValueError("Data Directory can not be invalid!")
                
                for sub_dir in os.listdir(data_dir):
                    sub_dir_path = os.path.join(data_dir, sub_dir)
                    img_paths_ = glob(f'{sub_dir_path}/*.{self._cfg.IMAGE_SUFFIX}')

                    img_paths.extend(img_paths_)
                    labels.extend(np.full(len(img_paths_), int(ft != 'original')))
                
        print('{} image paths have been loaded from FF++!'.format(len(img_paths)))          
        return img_paths, labels, mask_paths, ot_props
