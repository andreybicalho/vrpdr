import torch
import logging
import numpy as np

from torchvision import transforms

from attention_ocr.utils.tokenizer import Tokenizer
from attention_ocr.model.attention_ocr import AttentionOCR

class OCR():
    """
    Optical Character Recognition (OCR) aims to recognize characters from images.
    """

    def __init__(self, model_filename="config/attention_ocr_model.pth", use_cuda=False, n_chars=7):
        self.img_width = 160
        self.img_height = 60
        self.nh = 512

        self.chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

        self.img_trans = transforms.Compose([
             transforms.ToPILImage()
            ,transforms.Resize((self.img_height, self.img_width))
            ,transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])        

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        logging.debug(f'Using {self.device} device for Attention OCR.')

        self.tokenizer = Tokenizer(self.chars)
        self.model = AttentionOCR(self.img_width, self.img_height, self.nh, self.tokenizer.n_token,
                n_chars + 1, self.tokenizer.SOS_token, self.tokenizer.EOS_token).to(device=self.device)

        logging.debug("Loading Attention OCR...")
        self.model.load_state_dict(torch.load(model_filename, map_location=self.device))
        self.model.eval()

    def predict(self, input_img):
        """
        input_img: 3 channels (h,w,c) rgb image
        """
        t = self.img_trans(input_img)
        with torch.no_grad():
            pred = self.model(t.unsqueeze(0))
    
        result = self.tokenizer.translate(pred.squeeze(0).argmax(1))
        return result
        