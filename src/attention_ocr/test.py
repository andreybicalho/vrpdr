import torch

from model.attention_ocr import AttentionOCR
from utils.tokenizer import Tokenizer
from utils.dataset import SSIGALPRDataset
from utils.img_util import display_images

MODEL_PATH_FILE = '../../config/attention_ocr_model.pth'
ROOT_IMG_PATH = 'ssigalpr_samples/val/'
ANNOTADED_FILE = 'ssigalpr_samples/val.csv'

img_width = 160
img_height = 60
nh = 512
chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
print(f'char size: {len(chars)}')
n_chars = 7
device = 'cpu'

tokenizer = Tokenizer(chars)
model = AttentionOCR(img_width, img_height, nh, tokenizer.n_token,
                n_chars + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)

model.load_state_dict(torch.load(MODEL_PATH_FILE))

dataset = SSIGALPRDataset(img_width, img_height, n_chars=n_chars, labels_path=ANNOTADED_FILE, root_img_dir=ROOT_IMG_PATH)

img, label = dataset[0]
print(f'tensor shape: {img.shape}')
print(f'unsqueezed tensor shape: {img.unsqueeze(0).shape}')

model.eval()
with torch.no_grad():
    pred = model(img.unsqueeze(0))

print(f'pred shape: {pred.shape}') 
print(f'pred squeeze shape: {pred.squeeze(0).shape}')
print(f'pred squeeze argmax: {pred.squeeze(0).argmax(1)}')
pred = tokenizer.translate(pred.squeeze(0).argmax(1))
print(f'groundtruth: {tokenizer.translate(label)}\nprediction: {pred}')

display_images(img.numpy(), 1, 3)
