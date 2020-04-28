import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import argparse
import logging

from models import EMNISTNet

def plot_images(data, rows, cols, cmap='gray'):
    if(len(data) > 0):
        i = 0
        for title, image in data.items():
            #logging.debug(title)    
            plt.subplot(rows,cols,i+1),plt.imshow(image,cmap)
            plt.title(title)
            plt.xticks([]),plt.yticks([])
            i += 1
        plt.show()

def display_images(img_list, row, col):
    if(len(img_list) > 0):
        images = {}
        n = 0
        for img in img_list:
            n += 1
            images[str(n)] = img
        plot_images(images, row, col, cmap='gray')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_dataset(path_filename):
    if path_filename is not None:
        dataset = torchvision.datasets.ImageFolder(root=path_filename, 
                        transform=transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor()
                        ])   
                    )
        logging.debug(f'Training on {path_filename} dataset, size: {len(dataset)}')
    else :
        dataset = torchvision.datasets.EMNIST(
            root = 'data/',
            split='bymerge',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()                                 
            ])
        )        
        logging.debug(f'Training on EMNIST (bymerge) dataset, size: {len(dataset)}')

    data_loader = DataLoader(dataset, batch_size=1000, shuffle=True)
    batch = next(iter(data_loader))
    images, labels = batch
    num_classes = int(torch.max(labels)) + 1
    logging.debug(f'Number of classes: {num_classes}')

    return dataset, num_classes

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Usage: python test.py --m=result_model.pt --v')
    parser.add_argument('--m', help='Path to model')
    parser.add_argument('--v', type=str2bool, nargs='?', const=True, default=False, help='verbose and debug msgs')
    parser.add_argument('--d', help='dataset to test on')
    args = parser.parse_args()
    
    if args.v:
        DEBUG = True
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug('Verbose mode is activated.')

    test_set, num_classes = load_dataset(args.d)

    classes = ['0','1','2','3','4','5','6','7','8','9',
       'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
       'a','b','d','e','f','g','h','n','q','r','t']    

    NUM_IMAGES = 20

    loader = DataLoader(test_set, batch_size=NUM_IMAGES, shuffle=True)

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

    net = EMNISTNet(num_classes=num_classes)
    net.load_state_dict(torch.load(args.m))

    preds = net(images)
    preds = preds.argmax(dim=1)
    logging.info(f'preds: {preds.shape}\n{preds}')
    preds_indexes =  list(np.array(preds.squeeze(0)).astype(int))
    preds_classes_name = [classes[idx] for idx in preds_indexes]
    logging.debug(f'groundtruth classes: {groundtruth_classes_name}')
    logging.debug(f'preds_classes      : {preds_classes_name}')

    images = [ images[idx][0].numpy() for idx in range(NUM_IMAGES)]
    display_images(images, 1, NUM_IMAGES)
