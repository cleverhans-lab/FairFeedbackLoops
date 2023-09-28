import os

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as torch_utils
import torchvision.utils as torch_utils2


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr,
                            np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                            arr,
                            np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr


class CelebA(datasets.VisionDataset):
    """
    CelebA preparation. Predicting attractiveness from gender. 
    """
    def __init__(self, root='datasets/celeba_pytorch', train=True, valid=False, transform=None, target_transform=None, **kwargs):
        super(CelebA, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.root = root
        if not os.path.isfile('./data/celeba/celeba-gender-train.csv'):
            self.prep_csvs()

        if train:
            self.csv = './data/celeba/celeba-gender-train.csv'
        elif valid:
            self.csv = './data/celeba/celeba-gender-valid.csv'
        else:
            self.csv = './data/celeba/celeba-gender-test.csv'
        df = pd.read_csv(self.csv, index_col=0)
        self.sensitive = df['Male']
        self.label = df['Attractive']
        self.image_names = df.index.values
        self.imdir = os.path.join(os.path.join(root, 'celeba'), 'img_align_celeba')
        self.transform = transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, sensitive 20, target 2) where sensitive is fair attr and target is index of the target class.
        """
        img = Image.open(os.path.join(self.imdir, self.image_names[index]))
        if self.transform is not None:
            img = self.transform(img)
        
        lab = self.label[index]
        sens = self.sensitive[index]
        return img, sens, lab

    def __len__(self):
        return len(self.image_names)

    def prep_csvs(self):
        root_csv = os.path.join(os.path.join(self.root, 'celeba'), 'list_attr_celeba.txt')
        labels = pd.read_csv(root_csv, sep="\s+", skiprows=1, usecols=['Male', 'Attractive'])
        # Make 0 (female) & 1 (male) labels instead of -1 & 1
        labels.loc[labels['Male'] == -1, 'Male'] = 0
        labels.loc[labels['Attractive'] == -1, 'Attractive'] = 0

        csv_path = os.path.join(os.path.join(self.root, 'celeba'), 'list_eval_partition.txt')
        partition = pd.read_csv(csv_path, sep="\s+", skiprows=0, header=None)
        partition.columns = ['Filename', 'Partition']
        partition = partition.set_index('Filename')

        df3 = labels.merge(partition, left_index=True, right_index=True)
        df3.to_csv('./data/celeba/celeba-gender-partitions.csv')

        # split dataset
        df3.loc[df3['Partition'] == 0].to_csv('./data/celeba/celeba-gender-train.csv')
        df3.loc[df3['Partition'] == 1].to_csv('./data/celeba/celeba-gender-valid.csv')
        df3.loc[df3['Partition'] == 2].to_csv('./data/celeba/celeba-gender-test.csv')

                
def plot_dataset_images(dataset, save_as):
    images = []
    for i in range(50):
        img, sens, lab = dataset[i]
        images.append(img)

    np_imagegrid = torch_utils2.make_grid(images[1:50], 10, 5).numpy()
    recon_grid = np.transpose(np_imagegrid, (1, 2, 0))
    plt.imshow(recon_grid)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_as, dpi=400)
    plt.cla()
    plt.clf()
    plt.close()  


if __name__ == '__main__':

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize((128, 128)),
                                        transforms.ToTensor()])
    data = CelebA(train=True, valid=False, transform=transform)
    # transform = None #transforms.Compose([transforms.ToTensor()])
    # transform = transforms.Compose([transforms.CenterCrop(148),
    #                                     transforms.Resize((128, 128)),
    #                                     transforms.ToTensor()])
    # # train_set = CelebA(train=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
    # # print(len(train_set.data_label_tuples))
    # plot_dataset_images(train_loader, './figs/celeba_train.pdf')
    # plot_dataset_digits(test_set, './figs/cmnist_test.pdf')

    # data = CelebA(train=True, transform=transform)
    plot_dataset_images(data, './figs/celeba_train.pdf')
    

