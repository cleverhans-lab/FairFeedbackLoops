# adapted from https://colab.research.google.com/github/reiinakano/invariant-risk-minimization/blob/master/invariant_risk_minimization_colored_mnist.ipynb#scrollTo=sopHPgEhu4Jo CITED

import os

import numpy as np
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


class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Modifications: Labels 0 and 1, where 1 corresponds to benficial class. Colors are green and red, where
    green samples recieve benefit. Green_probas is likelihood sample is green given classes. For unfair training
    set, green more likeley in class 1. Use balanced [.5, .5] for test set

    Args:
        root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
        transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """
    def __init__(self, green_probas, root='./data/ColoredMNIST', train=True, valid=False, transform=None, target_transform=None, pos_class_thresh=5):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.root = root

        self.prepare_colored_mnist(green_probas=green_probas, is_train=train, is_valid=valid, pos_class_thresh=pos_class_thresh)
        
        assert not (train and valid)
        if train:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'train.pt'))
        elif not train and not valid:  # test
            self.data_label_tuples = torch.load(os.path.join(self.root, 'test.pt'))
        else:  # valid set
            self.data_label_tuples = torch.load(os.path.join(self.root, 'valid.pt'))


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

    def prepare_colored_mnist(self, green_probas=[.5, .5], is_train=True, is_valid=False, pos_class_thresh=5):
        """
        Train is 50000 and valid 10000 from same distr. Test is 10000 from balanced distr.
        """
        colored_mnist_dir = self.root
        if is_train or is_valid and os.path.exists(os.path.join(colored_mnist_dir, 'train.pt')) and os.path.exists(os.path.join(colored_mnist_dir, 'valid.pt')):
            print('Colored MNIST train and valid datasets already exists')
            # return
        if not is_train and not is_valid and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
            print('Colored MNIST test dataset already exists')
            # return
        
        print(f'Preparing Colored MNIST Train={is_train} Set')
        train_mnist = datasets.mnist.MNIST(self.root, train=is_train, download=True)
        print(green_probas)
        train_set = []
        valid_set = []

        labels = []
        fairs = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)
            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < pos_class_thresh else 1  # for label imbalance.

            # label noise at 5% or so
            if np.random.uniform() < 0.05:
                binary_label = binary_label ^ 1

            # induce relationship between color and label
            if binary_label == 0:
                if np.random.uniform() < green_probas[0]: #.7
                    color_red = 0 # green
                else:
                    color_red = 1 # red
            else:
                if np.random.uniform() < green_probas[1]: #.3
                    color_red = 0 # green
                else:
                    color_red = 1 # red

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            labels.append(binary_label)
            fairs.append(color_red)

            if idx > 50000:
                valid_set.append((Image.fromarray(colored_arr), color_red, binary_label))
            else:
                train_set.append((Image.fromarray(colored_arr), color_red, binary_label))

            # Debug
            # print('original label', type(label), label)
            # print('binary label', binary_label)
            # print('assigned color', 'red' if color_red else 'green')
            # plt.imshow(colored_arr)
            # plt.show()
            # break
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


        print(f'\nProportion red: {fairs.sum() / len(fairs)}')
        print(f'Proportion class 1 {labels.sum() / len(labels)}')
        print(f'red_pos {red_pos / len(fairs)} \nred_neg {red_neg / len(fairs)} \ngreen_pos {green_pos / len(fairs)} \ngreen_neg {green_neg / len(fairs)}\n')

        # if not os.path.exists(colored_mnist_dir):
        #     os.makedirs(colored_mnist_dir)
        # if is_train:
        #     torch.save(train_set, os.path.join(colored_mnist_dir, 'train.pt'))
        #     torch.save(valid_set, os.path.join(colored_mnist_dir, 'valid.pt'))
        # else:
        #     torch.save(train_set, os.path.join(colored_mnist_dir, 'test.pt'))



def plot_dataset_digits(dataset, save_as):
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
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = ColoredMNIST([.3, .7], train=True, transform=transform)
    # print(len(train_set.data_label_tuples))
    # print(train_set[0][0][0])  # vals between 0 and 1
    plot_dataset_digits(train_set, './figs/cmnist_train.pdf')
    # test_set = ColoredMNIST([.5, .5], train=False, transform=transform)
    # plot_dataset_digits(test_set, './figs/cmnist_test.pdf')

