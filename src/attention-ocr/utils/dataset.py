import os

from captcha.image import ImageCaptcha

from Crypto import Random
from Crypto.Random import random
from PIL import Image, ImageOps
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler

from utils.tokenizer import Tokenizer


img_trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=3)
    ,transforms.ToTensor()
    ,transforms.Normalize(mean=[0.5, 0.5, 0.5], std=(0.5, 0.5, 0.5))
])


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

        return img_trans(d), label

