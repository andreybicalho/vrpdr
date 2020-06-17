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

    model.load_state_dict(torch.load('./chkpoint/time_2020-06-16_19-49-27_epoch_95.pth'))

    img_trans = transforms.Compose([
        transforms.ToPILImage()
        ,transforms.Resize((img_height, img_width))
        ,transforms.Grayscale(num_output_channels=3)
        ,transforms.ToTensor()
        ,lambda x: x < 0.7 # thresholding (for '<' operator input img should have white background)
        ,lambda x: x.float() 
    ])

    if hasFrame:
        print(f'Frame shape: {frame.shape}')
        t = img_trans(frame)
        print(f'tensor shape: {t.shape}')
        print(f'unsqueezed tensor shape: {t.unsqueeze(0).shape}')

        model.eval()
        with torch.no_grad():            
            pred = model(t.unsqueeze(0))
    
        rst = tokenizer.translate(pred.squeeze(0).argmax(1))
        print(rst)

        display_images(t.numpy(), 1, 3)
    else:
        print("Frame not found!")