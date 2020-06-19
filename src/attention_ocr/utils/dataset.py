import numpy as np
import os
import pandas as pd

from captcha.image import ImageCaptcha

from Crypto import Random
from Crypto.Random import random
from PIL import Image, ImageOps

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler

from utils.tokenizer import Tokenizer

class CaptchaDataset(Dataset):
    def __init__(self, img_width, img_height, ds_size, n_chars=4, chars=None):
        self.gen = ImageCaptcha(img_width, img_height)
        self.size = ds_size

        self.n_chars = n_chars

        if chars is None:
            self.chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        else:
            self.chars = list(chars)

        self.tokenizer = Tokenizer(self.chars)

        self.first_run = True

        self.img_trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=3)
            ,transforms.ToTensor()
            ,lambda x: x < 0.7
            ,lambda x: x.float() 
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if self.first_run:
            Random.atfork()
            self.first_run = False

        content = [random.randrange(0, len(self.chars)) for _ in range(self.n_chars)]

        s = ''.join([self.chars[i] for i in content])

        d = self.gen.generate(s)
        d = Image.open(d)

        label = torch.full((self.n_chars + 2, ), self.tokenizer.EOS_token, dtype=torch.long)

        ts = self.tokenizer.tokenize(s)
        label[:ts.shape[0]] = torch.tensor(ts)

        return self.img_trans(d), label

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
            ,transforms.Grayscale(num_output_channels=3)
            ,transforms.ToTensor()
            ,transforms.Normalize(mean=[0.5, 0.5, 0.5], std=(0.5, 0.5, 0.5)) 
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
