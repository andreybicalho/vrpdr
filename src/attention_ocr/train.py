import argparse
import random
import time
import pickle

from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from model.attention_ocr import AttentionOCR
from utils.dataset import SSIGALPRDataset
from utils.train_util import train_batch, eval_batch

ROOT_TRAIN_IMG_DIR = 'F:\\dev\\ssigalpr_dataset\\test_train'
ANNOTADED_TRAIN_FILE = 'ssigalpr_samples/test_train.csv'
ROOT_VAL_IMG_DIR = 'F:\\dev\\ssigalpr_dataset\\val'
ANNOTADED_VAL_FILE = 'ssigalpr_samples/val.csv'


def main(inception_model='./inception_v3_google-1a9a5a14.pth', n_epoch=100, max_len=4, batch_size=32, n_works=4, 
         save_checkpoint_every=5, device='cuda', train_labels_path=ANNOTADED_TRAIN_FILE, train_root_img_dir=ROOT_TRAIN_IMG_DIR, 
         test_labels_path=ANNOTADED_VAL_FILE, test_root_img_dir=ROOT_VAL_IMG_DIR):
    img_width = 160
    img_height = 60
    nh = 512

    teacher_forcing_ratio = 0.5    
    lr = 3e-4    

    ds_train = SSIGALPRDataset(img_width, img_height, n_chars=7, labels_path=train_labels_path, root_img_dir=train_root_img_dir)
    ds_test = SSIGALPRDataset(img_width, img_height, n_chars=7, labels_path=test_labels_path, root_img_dir=test_root_img_dir)

    tokenizer = ds_train.tokenizer

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=n_works)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=n_works)

    model = AttentionOCR(img_width, img_height, nh, tokenizer.n_token,
                max_len + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=DEVICE)

    load_weights = torch.load(inception_model)

    names = set()
    for k, w in model.incept.named_children():
        names.add(k)

    weights = {}
    for k, w in load_weights.items():
        if k.split('.')[0] in names:
            weights[k] = w

    model.incept.load_state_dict(weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    crit = nn.NLLLoss().cuda()

    def train_epoch():
        sum_loss_train = 0
        n_train = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, batch in enumerate(tqdm(train_loader)):
            x, y = batch
            x = x.to(device=device)
            y = y.to(device=device)

            loss, acc, sentence_acc = train_batch(x, y, model, optimizer,
                                                  crit, teacher_forcing_ratio, max_len,
                                                  tokenizer)

            sum_loss_train += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_train += 1

        return sum_loss_train / n_train, sum_acc / n_train, sum_sentence_acc / n_train

    def eval_epoch():
        sum_loss_eval = 0
        n_eval = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, batch in enumerate(tqdm(test_loader)):
            x, y = batch
            x = x.to(device=device)
            y = y.to(device=device)

            loss, acc, sentence_acc = eval_batch(x, y, model, crit, max_len, tokenizer)

            sum_loss_eval += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_eval += 1

        return sum_loss_eval / n_eval, sum_acc / n_eval, sum_sentence_acc / n_eval

    for epoch in range(n_epoch):
        print("\nEpoch %d" % epoch)

        train_loss, train_acc, train_sentence_acc = train_epoch()
        print('train_loss: %.4f, train_acc: %.4f, train_sentence: %.4f' % (train_loss, train_acc, train_sentence_acc))

        eval_loss, eval_acc, eval_sentence_acc = eval_epoch()
        print('eval_loss:  %.4f, eval_acc:  %.4f, eval_sentence:  %.4f' % (eval_loss, eval_acc, eval_sentence_acc))

        if epoch % save_checkpoint_every == 0 and epoch > 0:
            print('saving checkpoint...')
            torch.save(model.state_dict(), './chkpoint/time_%s_epoch_%s.pth' % (time.strftime('%Y-%m-%d_%H-%M-%S'), epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python train.py --inception=\'./inception_v3_google-1a9a5a14.pth\' --e=1 --cuda')
    parser.add_argument('--e', type=int, nargs='?', const=100, default=100, help='Number of epochs to train the model')
    parser.add_argument('--l', type=int, nargs='?', const=7, default=7, help='Max number of characters in the image')
    parser.add_argument('--c', type=int, nargs='?', const=5, default=5, help='Save model every given number of epochs (checkpoint)')
    parser.add_argument('--w', type=int, nargs='?', const=4, default=4, help='Number of workers')
    parser.add_argument('--b', type=int, nargs='?', const=32, default=32, help='Batch size')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA')
    parser.add_argument('--inception', help='Path to the inception model')
    args = parser.parse_args()

    # TODO: 1) put all this params in a config file
    # 2) load previous model and continue from there
    NUM_EPOCHS = args.e if args.e is not None else 100
    MAX_LEN = args.l if args.l is not None else 7
    CHECKPOINT = args.c if args.c is not None else 5
    DEVICE = 'cuda' if args.cuda else 'cpu'
    N_WORKERS = args.w if args.w is not None else 6
    INCEPTION_MODEL = args.inception if args.inception is not None else './inception_v3_google-1a9a5a14.pth'
    BATCH_SIZE = args.b if args is not None else 32

    print(f'Device: {DEVICE} {args.cuda}\nEpochs: {NUM_EPOCHS}\nChar length: {MAX_LEN}')
    print(f'Checkpoint every: {CHECKPOINT} epochs\nNumber of workers: {N_WORKERS}')
    print(f'Batch size: {BATCH_SIZE}\nInception model: {INCEPTION_MODEL}')

    main(inception_model=INCEPTION_MODEL, n_epoch=NUM_EPOCHS, max_len=MAX_LEN, n_works=N_WORKERS, 
        save_checkpoint_every=CHECKPOINT, device=DEVICE, 
        train_labels_path=ANNOTADED_TRAIN_FILE, train_root_img_dir=ROOT_TRAIN_IMG_DIR,
        test_labels_path=ANNOTADED_VAL_FILE, test_root_img_dir=ROOT_VAL_IMG_DIR,
        batch_size=BATCH_SIZE)
