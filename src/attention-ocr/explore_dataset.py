import random
from PIL import Image

from captcha.image import ImageCaptcha
from utils.dataset import CaptchaDataset
from utils.img_util import display_images

from torchvision import transforms

import numpy as np

img_width = 160
img_height = 60
n_chars = 7

chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
gen = ImageCaptcha(img_width, img_height)


#img_trans = transforms.Compose([
#    transforms.Grayscale(num_output_channels=1)
#    ,transforms.ToTensor()
#    ,transforms.Normalize(mean=[0.5], std=[0.5])
##])

img_trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=3)
    ,transforms.ToTensor()
    ,transforms.Normalize(mean=[0.5, 0.5, 0.5], std=(0.5, 0.5, 0.5))
])

content = [random.randrange(0, len(chars)) for _ in range(n_chars)]
s = ''.join([chars[i] for i in content])
d = gen.generate(s)
d = Image.open(d)

t = img_trans(d)
print(f'\ntensor shape{t.shape}')


display_images(t.numpy(), 1, 3)