import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from collections import OrderedDict

import argparse
import logging

from pytorch_utils import TrainingManager, get_runs_params

from models import EMNISTNet, EMNISTNet_v2, EMNISTNet_v3, EMNISTNet_v4

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

logging.getLogger().setLevel(logging.INFO)
DEBUG = False

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
                            transforms.RandomApply([transforms.RandomAffine(degrees=(-20, 20), shear=(-30, 30))], p=0.5),
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

def load_net(net_version, pretrained_weights):
    if net_version is not None:
        if net_version == 1:
           net = EMNISTNet(num_classes=num_classes)
        elif net_version == 2:
           net = EMNISTNet_v2(num_classes=num_classes)
        elif net_version == 3:
           net = EMNISTNet_v3(num_classes=num_classes)
        elif net_version == 4:
           net = EMNISTNet_v4(num_classes=num_classes)
        else:
            net = EMNISTNet(num_classes=num_classes)
    else:
        net = EMNISTNet(num_classes=num_classes)

    if pretrained_weights is not None:
        logging.debug(f'Loading pre-trained model: {pretrained_weights}')
        net.load_state_dict(torch.load(pretrained_weights))

    return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python train_model.py --m=emnist_model.pt --d=custom_dataset/ --e=1 --cuda --v --o=emnist_model')
    parser.add_argument('--m', help='Path to a previous model to start with')
    parser.add_argument('--e', type=int, nargs='?', const=1, default=1, help='Number of epochs to train the model')
    parser.add_argument('--cuda', type=str2bool, nargs='?', const=False, default=False, help='use CUDA if available')
    parser.add_argument('--v', type=str2bool, nargs='?', const=False, default=False, help='verbose and debug msgs')
    parser.add_argument('--d', help='Path to the custom dataset')
    parser.add_argument('--o', help='Output model filename')
    parser.add_argument('--mobile', type=str2bool, nargs='?', const=False, default=False, help='export model for mobile loading')
    parser.add_argument('--n', type=int, nargs='?', const=1, default=1, help='net model to use')
    parser.add_argument('--b', type=int, nargs='?', const=500, default=500, help='batch size')
    args = parser.parse_args()
    
    if args.v:
        DEBUG = True
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug('Verbose mode is activated.')
        
    NUM_EPOCHS = args.e
    logging.debug(f'Number of epochs: {NUM_EPOCHS}')

    train_set, num_classes = load_dataset(args.d)

    # TODO: put run params in a config file or just remove multiple runs support?
    params = OrderedDict(
        lr = [0.001],
        batch_size = [args.b],
        shuffle = [True]
    )    

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f'Using device {device}')

    train_manager = TrainingManager()
    for run in get_runs_params(params):        
        net = load_net(args.n, args.m)
        net.to(device)

        data_loader = DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle)
        optimizer = optim.Adam(net.parameters(), lr=run.lr)

        train_manager.begin_run(run, net, data_loader)
        
        for epoch in range(NUM_EPOCHS):
            train_manager.begin_epoch()
            
            for batch in data_loader:                
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                preds = net(images)
                loss = F.cross_entropy(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_manager.track_loss(loss)
                train_manager.track_num_corret(preds, labels)

                logging.info(f'Loss: {loss}')

            train_manager.end_epoch()

        train_manager.end_run()

    if args.o is not None:
        train_manager.save(filename=args.o, export_for_mobile_loading=True)
