import numpy as np
import os
import pandas as pd

from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils.tokenizer import Tokenizer

class SSIGALPRDataset(Dataset):
    def __init__(self, img_width, img_height, n_chars=7, chars=None, labels_path='/path/to/the/annotated/file', root_img_dir='/path/to/img/dir'):
        self.n_chars = n_chars

        if chars is None:
            self.chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        else:
            self.chars = list(chars)

        self.tokenizer = Tokenizer(self.chars)

        df = pd.read_csv(labels_path, dtype={'img_id': str})
        self.annotaded_data = df.loc[df['text'] != 'no_one']
        self.root_img_dir = root_img_dir

        self.img_trans = transforms.Compose([
             transforms.Resize((img_height, img_width))
            ,transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
    
    def __len__(self):
        return self.annotaded_data.shape[0]

    def __getitem__(self, item):
        annotaded_item = self.annotaded_data.iloc[item]
        
        img_id = annotaded_item[0]
        img_path = self.root_img_dir + '/' + img_id + '.png'
        img = Image.open(img_path)

        width, height = img.size
        x0 = annotaded_item[1] * width
        y0 = annotaded_item[2] * height
        x1 = annotaded_item[3] * width
        y1 = annotaded_item[4] * height
        
        roi = img.crop((x0, y0, x1, y1))

        groundtruth = annotaded_item[5]
        groundtruth_label = torch.full((self.n_chars + 2, ), self.tokenizer.EOS_token, dtype=torch.long)
        ts = self.tokenizer.tokenize(groundtruth)
        groundtruth_label[:ts.shape[0]] = torch.tensor(ts)

        return self.img_trans(roi), groundtruth_label
