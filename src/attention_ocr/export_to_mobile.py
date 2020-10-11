import torch
import argparse

from torchvision import transforms

from utils.dataset import SSIGALPRDataset
from utils.tokenizer import Tokenizer

from model.attention_ocr import AttentionOCR

ROOT_IMG_PATH = 'ssigalpr_samples/val/'
ANNOTADED_FILE = 'ssigalpr_samples/val.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python export_to_mobile.py --m=chkpoint/time_2020-06-12_19-31-05_epoch_10.pth --out=exported_model.pth')
    parser.add_argument('--m', help='input model filename.')
    parser.add_argument('--out', help='output model filename.')
    parser.add_argument('--w', type=int, nargs='?', const=160, default=160, help='image width that the model was trained on.')
    parser.add_argument('--h', type=int, nargs='?', const=60, default=60, help='image height that the model was trained on.')
    args = parser.parse_args()

    img_width = args.w if args.w is not None else 160
    img_height = args.h if args.h is not None else 60
    nh = 512
    chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    n_chars = 7
    device = 'cpu'

    tokenizer = Tokenizer(chars)
    model = AttentionOCR(img_width, img_height, nh, tokenizer.n_token,
                n_chars + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)    

    model.load_state_dict(torch.load(args.m))

    dataset = SSIGALPRDataset(img_width, img_height, n_chars=n_chars, labels_path=ANNOTADED_FILE, root_img_dir=ROOT_IMG_PATH)

    img, label = dataset[0]

    input_img = img.unsqueeze(0)
    input_img.to(device)

    model = AttentionOCR(img_width, img_height, nh, tokenizer.n_token,
                n_chars + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)

    traced_cpu_model = torch.jit.trace(model, input_img)
    torch.jit.save(traced_cpu_model, f'{args.out}')
