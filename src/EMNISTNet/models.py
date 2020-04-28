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
        # in: 28x28x1
        x = F.relu(self.conv1(x))

        # in: 26x26x16
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # in: 12x12x64
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # fc
        # in: 5x5x64
        x = x.reshape(-1, 64*5*5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.out(x)
        
        return x

class EMNISTNet_v2(torch.nn.Module):
    def __init__(self, num_classes):
        super(EMNISTNet_v2, self).__init__()

        # conv
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.05)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv4_drop = nn.Dropout2d(p=0.05)

        # fc
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=64*4*2)
        self.fc2 = nn.Linear(in_features=64*4*2, out_features=64*4)
        self.fc2_drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(in_features=64*4, out_features=num_classes)
    
    def forward(self, x):        
        # output size = ((inputSize + 2*pad - filterSize) / stride) + 1
        # conv
        # in: 28x28x1
        x = F.relu(self.conv1(x))

        # in: 26x26x32
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        
        x = self.conv2_drop(x)

        # in: 12x12x64
        x = F.relu(self.conv3(x))

        # in: 10x10x64
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)

        x = self.conv4_drop(x)

        # fc
        # in: 4x4x64
        x = x.reshape(-1, 64*4*4)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc2_drop(x)

        x = self.out(x)

        return x

class EMNISTNet_v3(torch.nn.Module):
    def __init__(self, num_classes):
        super(EMNISTNet_v3, self).__init__()

        # conv
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)

        # fc
        self.fc1 = nn.Linear(in_features=64*1*1, out_features=64*4)
        self.fc2 = nn.Linear(in_features=64*4, out_features=64*4)
        self.out = nn.Linear(in_features=64*4, out_features=num_classes)
    
    def forward(self, x):        
        # output size = ((inputSize + 2*pad - filterSize) / stride) + 1
        # conv
        # in: 28x28x1
        x = F.relu(self.conv1(x))

        # in: 26x26x32
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        
        # in: 12x12x64
        x = F.relu(self.conv3(x))

        # in: 10x10x64
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)

        # in: 4x4x64
        x = F.relu(self.conv5(x))

        # in: 3x3x64
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=2, stride=2)

        # fc
        # in: 1x1x64
        x = x.reshape(-1, 64*1*1)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.out(x)

        return x

class EMNISTNet_v4(torch.nn.Module):
    def __init__(self, num_classes):
        super(EMNISTNet_v4, self).__init__()

        # conv
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        # fc
        self.fc1 = nn.Linear(in_features=128*7*7, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc2_drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(in_features=128, out_features=num_classes)
    
    def forward(self, x):        
        # output size = ((inputSize + 2*pad - filterSize) / stride) + 1
        # conv
        # in: 28x28x1
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)

        # in: 14x14x64
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        
        # fc
        # in: 7x7x128
        x = x.reshape(-1, 128*7*7)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc2_drop(x)

        x = self.out(x)

        return x
