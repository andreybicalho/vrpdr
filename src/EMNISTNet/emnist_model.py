import torch
import torch.nn as nn
import torch.nn.functional as F

class EMNISTNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(EMNISTNet, self).__init__()

        # conv output size = ((inputSize + 2*pad - filterSize) / stride) + 1
        # max pool with filterSize = 2 and stride = 2 shrinks down by half
        #      
        # input:  28x28x1
        # output: (((28 + 2*0 - 3) / 1) + 1) = (28 - 3) + 1 = 26 --> 26x26x16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        # input:  26x26x16
        # output: (26 - 3) + 1 = 24
        # max pool: floor(24/2) = 12 --> 12x12x64
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        # input:  12x12x64
        # output: (12 - 3) + 1 = 10 
        # max pool: floor(10/2) = 5 --> 5x5x64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        # fc
        # input: 5x5x64 --> flatten = 1600
        self.fc1 = nn.Linear(in_features=64*5*5, out_features=64*5*5)
        self.fc2 = nn.Linear(in_features=64*5*5, out_features=64*4*2)
        self.out = nn.Linear(in_features=64*4*2, out_features=num_classes)

    def forward(self, x):        
        # conv
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # fc
        x = x.reshape(-1, 64*5*5)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.out(x)
        
        return x
        