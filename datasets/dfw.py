#-*- coding: utf-8 -*-
import os
import numpy as np
from glob import glob

from .common import CommonDataset
from .builder import DATASETS


@DATASETS.register_module()
class DFW(CommonDataset):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
    def _load_from_path(self, split):
        print(self._cfg.DATA[self.split.upper()].ROOT)
        print(os.path.exists(self._cfg.DATA[self.split.upper()].ROOT))
        assert os.path.exists(self._cfg.DATA[self.split.upper()].ROOT), "Root path to dataset can not be None!"
        data = self._cfg["DATA"]
        data_type = data.TYPE
        fake_types = self._cfg.DATA[split.upper()]["FAKETYPE"]
        img_paths, labels, mask_paths, ot_props = [], [], [], []

        # Load image data for each type of fake techniques
        for idx, ft in enumerate(fake_types):
            data_dir = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, self.split, data_type, ft)
            print(data_dir)
            if not os.path.exists(data_dir):
                raise ValueError("Data Directory can not be invalid!")
            
            for sub_dir in os.listdir(data_dir):
                if sub_dir == 'desktop.ini': continue
                sub_dir_path = os.path.join(data_dir, sub_dir)
                for sub2_dir in os.listdir(sub_dir_path):
                    if sub2_dir == 'desktop.ini': continue
                    sub2_dir_path = os.path.join(sub_dir_path, sub2_dir)
                    for sub3_dir in os.listdir(sub2_dir_path):
                        if sub3_dir == 'desktop.ini': continue
                        sub3_dir_path = os.path.join(sub2_dir_path, sub3_dir)
                        # print(sub3_dir_path)
                        img_paths_ = glob(f'{sub3_dir_path}/*.{self._cfg.IMAGE_SUFFIX}')
                    
                # # sub_dir_path = data_dir
                # # img_paths_ = glob(f'{sub_dir_path}/*.{self._cfg.IMAGE_SUFFIX}')

                        img_paths.extend(img_paths_)
                        labels.extend(np.full(len(img_paths_), int(ft == 'fake_test')))
                
        print('{} image paths have been loaded from DFW!'.format(len(img_paths)))          
        return img_paths, labels, mask_paths, ot_props
