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

train_data = torchvision.datasets.EMNIST(
        root = 'data/',
        split='bymerge',
        train = True,
        download = True,
        transform=transforms.Compose([
            transforms.ToTensor()
            ])
    )

NUM_IMAGES = 10

groundtruth = ['0','1','2','3','4','5','6','7','8','9',
              'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
              'a','b','d','e','f','g','h','n','q','r','t']

data_loader = torch.utils.data.DataLoader(train_data, batch_size=NUM_IMAGES, shuffle=True)
batch = next(iter(data_loader))
print(f'batch len: {len(batch)}')
print(f'type: {type(batch)}')
images, labels = batch
print(f'images shape: {images.shape}')
print(f'labels shape: {labels.shape}')
print(f'labels: {labels}')
print(f'pixels type:\n {type(images[0][0][0][0])}')
print(f'pixels max and min values:\n {torch.max(images[0][0])} and {torch.min(images[0][0])}')
#print(f'pixels:\n {images[0][0]}')
# plotting images
#grid = torchvision.utils.make_grid(images, nrow=NUM_IMAGES)
#plt.figure(figsize=(15,15))
#plt.imshow(np.transpose(grid, (1,2,0)))
#plt.show()

groundtruth_labels_indexes = list(np.array(labels.squeeze(0)).astype(int))
groundtruth_classes_name = [groundtruth[idx] for idx in groundtruth_labels_indexes]
print(f'groundtruth classes: {groundtruth_classes_name}')

# plotting images
images = [ images[idx][0].numpy() for idx in range(NUM_IMAGES)]
display_images(images, 2, 5)