import torch.utils.data
import os
import random
import torch
import numpy as np
import lmdb
import io
from PIL import Image

class LMDBLoader():
    def __init__(self, lmdb_path, num_sample):
        self.lmdb_env = lmdb.open(lmdb_path, readonly=True)
        self.lmdb_txn = self.lmdb_env.begin()
        self.lmdb_cursor = self.lmdb_txn.cursor()
        self.lmdb_cursor.first()
        self.num_sample = num_sample
        self.first_time = True
        
    def next(self):
        if not self.lmdb_cursor.next():
            self.lmdb_cursor.first()
    
    def _parse_rgb_lmdb(self):        
        data_list = []
        for _ in range(self.num_sample):
            data_list.append(self.lmdb_cursor.value())
            self.next()
        return data_list
    
    def _skip(self, skip_num):
        #for i in range(skip_num * self.num_sample):
        #    self.next()
        target_key = '{:0>10d}'.format(skip_num * self.num_sample).encode()
        value = self.lmdb_cursor.get(target_key, default='')
        if value == '':
            assert value != ''
        
class VideoMergeDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, video_num, repeat_num, transform, clip_length=1, num_steps=1, num_segments=1, num_channels=3):
        super(VideoMergeDataset, self).__init__()
        self.lmdb_path = lmdb_path
        self.video_num = video_num
        self.repeat_num = repeat_num
        self.transform = transform
        self.clip_length = clip_length
        self.num_steps = num_steps
        self.num_segments = num_segments
        self.num_channels = num_channels
        self.lmdb_loader = LMDBLoader(lmdb_path, num_segments * clip_length)
                
    def __len__(self):
        return self.video_num

    def __getitem__(self, item):
        if self.lmdb_loader.first_time:
            worker_info = torch.utils.data.get_worker_info()
            self.lmdb_loader._skip((worker_info.id * self.video_num + item) * self.repeat_num // worker_info.num_workers)           
            self.lmdb_loader.first_time = False
            
        data_list = self.lmdb_loader._parse_rgb_lmdb()

        image_list = []
        label = -1
        for data in data_list:
            if label < 0 :
                label = int.from_bytes(data[0:2], byteorder='big')
            else:
                assert label == int.from_bytes(data[0:2], byteorder='big')
            bio = io.BytesIO(data[2:])
            image = Image.open(bio).convert('RGB')
            image_list.append(image)
        
        trans_image_list = self.transform(image_list)
        return trans_image_list, label