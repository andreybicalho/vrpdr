import torch
import argparse
import random
from PIL import Image

from torchvision import transforms

from captcha.image import ImageCaptcha
from utils.tokenizer import Tokenizer

from model.attention_ocr import AttentionOCR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python export_to_mobile.py --in=time_2020-06-12_19-31-05_epoch_10.pth --out=exported_model.pth')
    parser.add_argument('--in', help='input model filename.')
    parser.add_argument('--out', help='output model filename.')
    parser.add_argument('--w', type=int, nargs='?', const=160, default=160, help='image width that the model was trained on.')
    parser.add_argument('--h', type=int, nargs='?', const=60, default=60, help='image height that the model was trained on.')
    args = parser.parse_args()

    device = "cpu"

    # generate some data in order to trace the model
    chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    gen = ImageCaptcha(args.w, args.h)
    n_chars = 7
    tokenizer = Tokenizer(chars)

    img_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3)
        ,transforms.ToTensor()
        ,transforms.Normalize(mean=[0.5, 0.5, 0.5], std=(0.5, 0.5, 0.5))
    ])

    content = [random.randrange(0, len(chars)) for _ in range(n_chars)]
    s = ''.join([chars[i] for i in content])
    d = gen.generate(s)
    d = Image.open(d)

    input_img = img_trans(d).unsqueeze(0)
    input_img.to(device)

    nh = 512
    model = AttentionOCR(args.w, args.h, nh, tokenizer.n_token,
                n_chars + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)

    traced_cpu_model = torch.jit.trace(model, input_img)
    torch.jit.save(traced_cpu_model, f'{args.out}')