import numpy as np
import os
import pandas as pd

from utils.dataset import SSIGALPRDataset
from utils.img_util import display_images

from torchvision import transforms

from PIL import Image

ANNOTADED_FILE = 'ssigalpr_samples/train.csv'
IMG_DIR = 'ssigalpr_samples/train/'

IMG_WIDTH = 160
IMG_HEIGHT = 60
N_CHARS = 7
CHARS = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')


img_trans = transforms.Compose([
     transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
    ,transforms.ToTensor()
    ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

if __name__ == '__main__':    
    df = pd.read_csv(ANNOTADED_FILE, dtype={'img_id': str})
    print(f'dataframe shape: {df.shape}')
    print(f'total items: {df.shape[0]}')

    df = df.loc[df['text'] != 'no_one']
    print(f'total items after cleaning: {df.shape[0]}')

    annotaded_data = df.iloc[0]

    img_id = annotaded_data.iloc[0]
    print(f'image: {img_id}.png')    
    img = Image.open(IMG_DIR+img_id+'.png')
    img.show()

    width, height = img.size
    x0 = annotaded_data.iloc[1] * width
    y0 = annotaded_data.iloc[2] * height
    x1 = annotaded_data.iloc[3] * width
    y1 = annotaded_data.iloc[4] * height

    label = annotaded_data.iloc[5]
    print(f'label: {label}')

    roi = img.crop((x0, y0, x1, y1))
    roi.show()

    t = img_trans(roi)
    display_images(t.numpy(), 1, 3)