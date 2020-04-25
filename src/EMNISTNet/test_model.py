import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import argparse
import logging

from emnist_model import EMNISTNet

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Usage: python test.py --m=result_model.pt --cuda --v')
    parser.add_argument('--m', help='Path to model')
    parser.add_argument('--cuda', type=str2bool, nargs='?', const=True, default=False, help='use CUDA if available')
    parser.add_argument('--v', type=str2bool, nargs='?', const=True, default=False, help='verbose and debug msgs')
    args = parser.parse_args()
    
    if args.v:
        DEBUG = True
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug('Verbose mode is activated.')

    test_set = torchvision.datasets.EMNIST(
        root = 'data/',
        split='bymerge',
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor()                                 
        ])
    )

    classes = ['0','1','2','3','4','5','6','7','8','9',
       'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
       'a','b','d','e','f','g','h','n','q','r','t']

    loader = DataLoader(test_set, batch_size=10, shuffle=True)

    batch = next(iter(loader))
    logging.debug(f'batch len: {len(batch)}')
    logging.debug(f'type: {type(batch)}')
    images, labels = batch
    logging.debug(f'images shape: {images.shape}')
    logging.debug(f'labels shape: {labels.shape}')
    
    logging.debug(f'labels: {labels}')
    groundtruth_labels_indexes = list(np.array(labels.squeeze(0)).astype(int))
    logging.debug(f'labels_indexes: {groundtruth_labels_indexes}')
    groundtruth_classes_name = [classes[idx] for idx in groundtruth_labels_indexes]
    logging.debug(f'groundtruth classes: {groundtruth_classes_name}')
    
    grid = torchvision.utils.make_grid(images, nrow=20)
    logging.debug(f'grid shape: {grid.shape}')
    plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f'Using device {device}')

    net = EMNISTNet(num_classes=47)
    net.load_state_dict(torch.load(args.m))
    net.to(device)
    preds = net(images)
    preds = preds.argmax(dim=1)
    logging.info(f'preds: {preds.shape}\n{preds}')
    preds_indexes =  list(np.array(preds.squeeze(0)).astype(int))
    preds_classes_name = [classes[idx] for idx in preds_indexes]
    logging.debug(f'preds_classes: {preds_classes_name}')
    logging.debug(f'groundtruth classes: {groundtruth_classes_name}')



