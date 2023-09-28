import os

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

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



def color_darken(arr, dark=True):
    """Darkens images for minoritized class"""
    image = Image.fromarray(arr)
    bright_factor = .5
    if dark:
        darkener = ImageEnhance.Brightness(image)
        image = darkener.enhance(bright_factor)
    return image


def color_to_gray(arr, dark=True):
    """Greyscales images for minoritized class"""
    image = Image.fromarray(arr)
    if dark:
        image = ImageOps.grayscale(image)  # deletes other channels
        image = np.array(image)
        image = np.expand_dims(image, 2)
        image = np.concatenate([image] * 3, axis=2)
    return image


def red_green(arr, dark=True):
    image = Image.fromarray(arr)
    image = ImageOps.grayscale(image)
    image = np.array(image)
    image = np.expand_dims(image, 2)
    h, w, c = image.shape
    dtype = image.dtype
    if dark:
        arr = np.concatenate([image,
                            np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                            image,
                            np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr


class SVHN(datasets.VisionDataset):
    """
    SVHN (like MNIST but more complex), with options above as the sensitive feature. As in https://arxiv.org/pdf/2103.00950.pdf
    Class 1 (the advantaged class is class 1, as MC converges to this class.)
    Task is to detect of greater of less than 5 (or some other threshold).

    Note that torch requires images of [c, w, h]

    Args:
        root (string): Root directory of dataset where ``SVHN/*.pt`` will exist.
        transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """
    def __init__(self, green_probas, root='./data/SVHN', split='train', transform=None, target_transform=None, pos_class_thresh=5, overwrite=0, seed=0):
        super(SVHN, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.root = root
        self.prepare_svhn(green_probas=green_probas, split=split, pos_class_thresh=pos_class_thresh, overwrite=overwrite, seed=seed)
        
        self.data_label_tuples = torch.load(os.path.join(self.root, f'{split}.pt'))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, sensitive, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, sensitive, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_svhn(self, green_probas=[.5, .5], split='train', pos_class_thresh=5, overwrite=0, seed=0):
        svhn_dir = self.root
        if overwrite == 0:
            if split in ['train', 'valid'] and os.path.exists(os.path.join(svhn_dir, 'train.pt')) and os.path.exists(os.path.join(svhn_dir, 'valid.pt')):
                print('SVHN train and valid datasets already exists')
                return
            if split == 'test' and os.path.exists(os.path.join(svhn_dir, 'test.pt')):
                print('SVHN test dataset already exists')
                return
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f'Preparing SVHN {split} Set')
        is_train = False
        if split in ['train', 'valid']:
            is_train = True
            train_mnist = datasets.SVHN(root=self.root, split='train', download=True)
        else:
            train_mnist = datasets.SVHN(root=self.root, split='test', download=True)
        print(green_probas)
        train_set = []
        valid_set = []

        labels = []
        fairs = []
        for idx, (im, label) in enumerate(train_mnist):
            im_array = np.array(im)

            # Class 1 is the large numbers
            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label >= pos_class_thresh else 1  # for label imbalance.

            # label noise at 5% or so
            if np.random.uniform() < 0.05:
                binary_label = binary_label ^ 1

            # induce relationship between color and label
            if binary_label == 0:  # disadv class
                if np.random.uniform() < green_probas[0]: # .3
                    color_red = 0 # bright/green
                else:
                    color_red = 1 # dark/gray/red
            else:  # adv class
                if np.random.uniform() < green_probas[1]: # .7
                    color_red = 0 # green/bright
                else:
                    color_red = 1 # dark/gray/red

            colored_arr = red_green(im_array, dark=color_red)  # darken images with color_red=1
            labels.append(binary_label)
            fairs.append(color_red)

            if idx > 52326:
                valid_set.append((colored_arr, color_red, binary_label))
            else:
                train_set.append((colored_arr, color_red, binary_label))

            # Check:
            # print('original label', type(label), label)
            # print('binary label', binary_label)
            # print('assigned color', 'red' if color_red else 'green')
            # plt.imshow(colored_arr)
            # plt.show()
        fairs = np.array(fairs)
        labels = np.array(labels)
        red_pos = 0
        red_neg = 0
        green_pos = 0
        green_neg = 0
        for i in range(len(fairs)):
            if fairs[i] == 1 and labels[i] == 1:  # red and pos
                red_pos += 1
            elif fairs[i] == 0 and labels[i] == 1:  # green and pos
                green_pos += 1
            elif fairs[i] == 0 and labels[i] == 0:  # green and neg
                green_neg += 1
            else:
                red_neg += 1


        # print(f'\nProportion dark: {fairs.sum() / len(fairs)}')
        # print(f'Proportion class 1 {labels.sum() / len(labels)}')
        # print(f'dark_pos {red_pos / len(fairs)} \ndark_neg {red_neg / len(fairs)} \nlight_pos {green_pos / len(fairs)} \nlight_neg {green_neg / len(fairs)}\n')

        if not os.path.exists(svhn_dir):
            os.makedirs(svhn_dir)
        if is_train:
            torch.save(train_set, os.path.join(svhn_dir, 'train.pt'))
            torch.save(valid_set, os.path.join(svhn_dir, 'valid.pt'))
        else:
            torch.save(train_set, os.path.join(svhn_dir, 'test.pt'))

        print(f'{fairs.sum() / len(fairs)}, {labels.sum() / len(labels)}, {red_pos / len(fairs)}, {red_neg / len(fairs)}, {green_pos / len(fairs)}, {green_neg / len(fairs)}')



def plot_dataset_digits(dataset, save_as):
    # fig = plt.figure(figsize=(13, 8))
    columns = 10
    rows = 5
    # ax enables access to manipulate each of subplots
    ax = []

    images = []
    for i in range(50):
        img, sens, lab = dataset[i]
        images.append(img)

    fig = plt.figure()
    fig.set_size_inches(w=6.4, h=4.81)
    np_imagegrid = torch_utils2.make_grid(images[1:50], 10, 5).numpy()
    recon_grid = np.transpose(np_imagegrid, (1, 2, 0))
    plt.imshow(recon_grid)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_as)
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(64, antialias=False)])
    train_set = SVHN([.3, .7], split='train', transform=transform, overwrite=0, seed=0)
    plot_dataset_digits(train_set, './figs/svhn_train.pdf')
    # print(len(train_set))
    # valid_set = SVHN([.3, .7], split='valid', transform=transform, overwrite=1)
    # print(len(valid_set))
    # test_set = SVHN([.5, .5], split='test', transform=transform, overwrite=1)
    # print(len(test_set))

    # print(f'Proportion minor,Proportion class 1,nmin_pos,nmin_neg,nmaj_pos,nmaj_neg')
    # for seed in range(25):
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)

    #     train_set = SVHN([.3, .7], split='train', transform=transform, overwrite=1)
        # plot_dataset_digits(train_set, './figs/svhn_train.pdf')
        # test_set = SVHN([.5, .5], split='test', transform=transform, overwrite=1)
        # plot_dataset_digits(test_set, './figs/svhn_test.pdf')

