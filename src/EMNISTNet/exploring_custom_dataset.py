import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

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

train_data = torchvision.datasets.ImageFolder(root='custom_dataset/', 
                transform=transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.RandomApply([transforms.RandomAffine(degrees=(-30, 30), shear=(-30, 30))], p=1.0),
                    transforms.ToTensor()
                ])
            )
print(f'dataset size: {len(train_data)}')

NUM_IMAGES = 36

groundtruth = ['0','1','2','3','4','5','6','7','8','9',
               'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

data_loader = torch.utils.data.DataLoader(train_data, batch_size=NUM_IMAGES, shuffle=True)
batch = next(iter(data_loader))
print(f'batch len: {len(batch)}')
print(f'type: {type(batch)}')
images, labels = batch
print(f'batch size: {len(images)}')
print(f'images shape: {images.shape}')
print(f'labels shape: {labels.shape}')
print(f'labels: {labels}')
print(f'pixels type:\n {type(images[0][0][0][0])}')
print(f'pixels max and min values:\n {torch.max(images[0][0])} and {torch.min(images[0][0])}')
print(f'pixels max and min values:\n {torch.max(images)} and {torch.min(images)}')

groundtruth_labels_indexes = list(np.array(labels.squeeze(0)).astype(int))
groundtruth_classes_name = [groundtruth[idx] for idx in groundtruth_labels_indexes]
print(f'groundtruth classes: {groundtruth_classes_name}')

# plotting images
images = [ images[idx][0].numpy() for idx in range(NUM_IMAGES)]
display_images(images, 1, 36)