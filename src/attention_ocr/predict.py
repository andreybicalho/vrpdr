import cv2 as cv
import torch
import numpy as np
import argparse
import sys
import os.path
import matplotlib.pyplot as plt

from torchvision import transforms

from model.attention_ocr import AttentionOCR
from utils.tokenizer import Tokenizer
from utils.img_util import display_images

MODEL_PATH_FILE = '../../config/attention_ocr_model.pth'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python predict.py --image=path/to/the/image/file.jpg')
    parser.add_argument('--image', help='Path to image file.')
    args = parser.parse_args()

    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)

    hasFrame, frame = cap.read()


    chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    img_width = 160
    img_height = 60
    nh = 512
    n_chars = 7
    device = "cpu"

    tokenizer = Tokenizer(chars)
    model = AttentionOCR(img_width, img_height, nh, tokenizer.n_token,
                n_chars + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)

    model.load_state_dict(torch.load(MODEL_PATH_FILE))

    img_trans = transforms.Compose([
        transforms.ToPILImage()
        ,transforms.Resize((img_height, img_width))
        ,transforms.ToTensor()
        ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    if hasFrame:
        print(f'Frame shape: {frame.shape}')
        img = img_trans(frame)
        print(f'tensor shape: {img.shape}')
        print(f'unsqueezed tensor shape: {img.unsqueeze(0).shape}')

        model.eval()
        with torch.no_grad():
            pred = model(img.unsqueeze(0))
    
        pred = tokenizer.translate(pred.squeeze(0).argmax(1))
        print(f'prediction: {pred}')

        display_images(img.numpy(), 1, 3)
    else:
        print("Frame not found!")
