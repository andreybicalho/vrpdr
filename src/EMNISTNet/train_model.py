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
DEBU = False
NUM_EPOCHS = 5

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
    parser = argparse.ArgumentParser(description='Usage: python train_model.py --m=emnist_model.pt --d=custom_dataset/ --e=1 --cuda --v --o=emnist')
    parser.add_argument('--m', help='Path to a previous model to start with')
    parser.add_argument('--e', help='Number of epochs to train the model')
    parser.add_argument('--cuda', type=str2bool, nargs='?', const=True, default=True, help='use CUDA if available')
    parser.add_argument('--v', type=str2bool, nargs='?', const=True, default=False, help='verbose and debug msgs')
    parser.add_argument('--d', help='Path to the custom dataset')
    parser.add_argument('--o', help='Output filename')
    parser.add_argument('--n', help='net model to use')
    parser.add_argument('--b', type=int, nargs='?', const=1000, default=500, help='batch size')
    args = parser.parse_args()
    
    if args.v:
        DEBUG = True
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug('Verbose mode is activated.')
        
    if args.e:
        NUM_EPOCHS = int(args.e)

    if args.d is not None:
        logging.debug(f'Training on custom dataset: {args.d}')
        train_set = torchvision.datasets.ImageFolder(root=args.d, 
                        transform=transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor()
                        ])   
                    )
    else :
        logging.debug('Training on EMNIST (bymerge) dataset.')
        train_set = torchvision.datasets.EMNIST(
            root = 'data/',
            split='bymerge',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()                                 
            ])
        )        

    # TODO: put run params in a config file or just remove multiple runs support?
    params = OrderedDict(
        lr = [0.01],
        batch_size = [args.b],
        shuffle = [True]
    )

    data_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
    batch = next(iter(data_loader))
    images, labels = batch
    num_classes = int(torch.max(labels)) + 1
    logging.debug(f'Number of classes: {num_classes}')

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f'Using device {device}')

    train_manager = TrainingManager()
    for run in get_runs_params(params):
        logging.info(f'Training model with the following parameters:\nlr: {run.lr}\nbatch_size: {run.batch_size}\nshuffle: {run.shuffle}')
        
        if args.n is not None:
            if args.n == 1:
                net = EMNISTNet(num_classes=num_classes)
            elif args.n == 2:
                net = EMNISTNet_v2(num_classes=num_classes)
            elif args.n == 3:
                net = EMNISTNet_v3(num_classes=num_classes)
            elif args.n == 4:
                net = EMNISTNet_v4(num_classes=num_classes)
            else:
                net = EMNISTNet(num_classes=num_classes)
        else:
            net = EMNISTNet(num_classes=num_classes)


        if args.m is not None:
            logging.debug(f'Loading pre-trained model: {args.m}')
            net.load_state_dict(torch.load(args.m))

        net.to(device)
        data_loader = DataLoader(train_set, batch_size=run.batch_size)
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
        train_manager.save(filename=args.o)
