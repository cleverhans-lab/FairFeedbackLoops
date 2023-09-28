import torch
import os
import copy
import numpy as np
import pandas as pd
import shutil
from scipy import stats
import collections
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import pynvml
import types
import ci_mnist
import prep_celeba
import svhn

import model


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=lambda x: x, ind=False):
        self.data = x
        self.labels = y
        self.transform = transform
        if ind:
            self.indices = np.arange(len(x))
        else:
            self.indices = None

    def __getitem__(self, index):
        x = self.transform(self.data[index]) if self.transform else self.data[index]
        y = self.labels[index]
        if self.indices is not None:
            return x, y, self.indices[index]
        else:
            return x, y

    def __len__(self):
        return self.data.shape[0]


def get_parameters(net, numpy=False, squeeze=True, trainable_only=True):
    trainable = []
    non_trainable = []
    trainable_name = [name for (name, param) in net.named_parameters()]
    state = net.state_dict()
    for i, name in enumerate(state.keys()):
        if name in trainable_name:
            trainable.append(state[name])
        else:
            non_trainable.append(state[name])

    if squeeze:
        trainable = torch.cat([i.reshape([-1]) for i in trainable])
        # print(non_trainable)
        if len(non_trainable) > 0:
            non_trainable = torch.cat([i.reshape([-1]) for i in non_trainable])
        if numpy:
            trainable = trainable.cpu().numpy()
            if len(non_trainable) > 0:
                non_trainable = non_trainable.cpu().numpy()

    if trainable_only:
        parameter = trainable
    else:
        parameter = trainable + non_trainable

    return parameter


def set_parameters(net, parameters, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                   verbose=False):
    net.load_state_dict(to_state_dict(net, parameters, device, verbose))
    return net


def to_state_dict(net, parameters, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                  verbose=False):
    state_dict = OrderedDict()
    trainable_name = [name for (name, param) in net.named_parameters()]
    if len(trainable_name) < len(parameters):
        if verbose:
            print("Setting trainable and non-trainable parameters")
        i, j = 0, 0
        for name in net.state_dict().keys():
            if name in trainable_name:
                if isinstance(parameters[i], torch.Tensor):
                    state_dict[name] = parameters[i].to(device)
                else:
                    state_dict[name] = torch.Tensor(parameters[i]).to(device)
                i += 1
            else:
                if isinstance(parameters[len(trainable_name) + j], torch.Tensor):
                    state_dict[name] = parameters[len(trainable_name) + j].to(device)
                else:
                    state_dict[name] = torch.Tensor(parameters[len(trainable_name) + j]).to(device)
                j += 1
    else:
        if verbose:
            print("Setting trainable parameters only")
        i = 0
        for name in net.state_dict().keys():
            if name in trainable_name:
                if isinstance(parameters[i], torch.Tensor):
                    state_dict[name] = parameters[i].to(device)
                else:
                    state_dict[name] = torch.Tensor(parameters[i]).to(device)
                i += 1
            else:
                state_dict[name] = net.state_dict()[name]
    return state_dict


def record_to_csv(data, file, headers=None):
    data = [str(x) for x in data]
    if os.path.isfile(file):
        # append data
        with open(file, 'a') as fo:
            fo.write(','.join(data)+'\n')
    else:
        # create file, write headers, write data
        assert len(data) == len(headers)
        with open(file, 'a+') as fo:
            fo.write(','.join(headers)+'\n'+','.join(data)+'\n')



def load_dataset(dataset, train, valid=False, download=False, numpy_data=None, apply_transform=True, green_probas=[.5, .5], pos_class_thresh=5, seed=0):
    if dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            if apply_transform:
                transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(), normalize])
            else:
                transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
            data = torchvision.datasets.ImageFolder("/scratch/ssd004/datasets/imagenet/train", transform=transform)
        else:
            transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
            data = torchvision.datasets.ImageFolder("/scratch/ssd004/datasets/imagenet/val", transform=transform)
    elif dataset == 'celeba':
        # transform = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                       transforms.CenterCrop(148),
        #                                       transforms.Resize((64, 64)),
        #                                       transforms.ToTensor()])
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Resize((128, 128))
        ])
        data = prep_celeba.CelebA(train=train, valid=valid, transform=transform)
    elif dataset == 'ColoredMNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        if train:
            dataset = ci_mnist.ColoredMNIST(green_probas, train=True, valid=False, transform=transform, pos_class_thresh=pos_class_thresh)
        elif not train and not valid:
            dataset = ci_mnist.ColoredMNIST(green_probas, train=False, valid=False, transform=transform, pos_class_thresh=pos_class_thresh)
        else: # valid
            dataset = ci_mnist.ColoredMNIST(green_probas, train=False, valid=True, transform=transform, pos_class_thresh=pos_class_thresh)
        data = dataset  
    elif dataset == 'gtsrb' or dataset == 'lisa':
        if train:
            np_data = np.load(f"data/{dataset}/x_train.npy")
            np_label = np.load(f"data/{dataset}/y_train.npy").astype(np.longlong)
        else:
            np_data = np.load(f"data/{dataset}/x_test.npy")
            np_label = np.load(f"data/{dataset}/y_test.npy").astype(np.longlong)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and apply_transform:
            transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize])
        else:
            transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), normalize])
        data = CustomTensorDataset(np_data, np_label, transform)
    elif dataset == 'ag_news':
        from torchtext.datasets import AG_NEWS
        from torchtext.data.functional import to_map_style_dataset

        if train:
            data = to_map_style_dataset(AG_NEWS(split='train'))
        else:
            data = to_map_style_dataset(AG_NEWS(split='test'))
    elif dataset == "SVHN":
        dataset_class = eval(f"torchvision.datasets.{dataset}")
        if train or valid:
            transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614]),
                transforms.Resize(64, antialias=False)])  
            if train:
                split = 'train'
            if valid:
                split = 'valid'
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(64, antialias=False)])
            split = 'test'
        data = svhn.SVHN(root=f'./data/SVHN_{seed}', green_probas=green_probas, split=split, transform=transform, pos_class_thresh=5, seed=seed)
    elif dataset == "yelp":
        from torchtext.datasets import YelpReviewFull
        if train:
            data = YelpReviewFull(split='train')
        else:
            data = YelpReviewFull(split='test')
    elif dataset == "sst2":
        from torchtext.datasets import SST2
        if train:
            data = SST2(split='train')
        else:
            data = SST2(split='test')
    else:
        try:
            dataset_class = eval(f"torchvision.datasets.{dataset}")
        except:
            # raise NotImplementedError(f"Dataset {dataset} is not implemented by pytorch.")
            pass

        if dataset == "MNIST":  
            transform = transforms.Compose([
                transforms.ToTensor()])
        elif dataset == "FashionMNIST":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,))])
        elif dataset == "CIFAR100":
            if train and apply_transform:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                    transforms.Resize([224, 224])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                    transforms.Resize([224, 224])
                ])
        else:
            if train and apply_transform:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if numpy_data is None:
            try:
                data = dataset_class(root='./data', train=train, download=download, transform=transform)
            except:
                if train:
                    data = dataset_class(root='./data', split="train", download=download, transform=transform)
                else:
                    data = dataset_class(root='./data', split="test", download=download, transform=transform)
        else:
            if dataset.startswith('mog'):
                numpy_data = (torch.tensor(numpy_data[0]).to(torch.float32), torch.tensor(numpy_data[1]).to(torch.int64))
                transform = None
            else:
                raise NotImplementedError(f"{dataset} not supported")
            data = CustomTensorDataset(numpy_data[0], numpy_data[1], transform)
    
    
    
    return data


def num_parameters(net):
    return sum(p.numel() for p in net.parameters())


def find_last_chekpoint(dir_name):
    list_checkpoints = [0]
    for d in os.listdir(dir_name):
        if d.startswith("model"):
            try:
                ckpt = int(d.split('.')[0][6:])
            except:
                raise ValueError('Unexpected error happened at loading previous checkpoints')
            list_checkpoints.append(ckpt)
    return max(list_checkpoints)


def get_optimizer(dataset, net, lr, num_batch, dec_lr=None, privacy_engine=None, gamma=0.1, optimizer="sgd", weight_decay=None):
    if dataset == 'MNIST' and optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    elif dataset == "celeba" and optimizer == "ADAM":
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
        scheduler = None
    elif dataset == 'CIFAR10' and optimizer == "sgd":
        if dec_lr is None:
            dec_lr = [100, 150]
        if gamma is None:
            gamma = 0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[round(i * num_batch) for i in dec_lr],
                                                   gamma=gamma)
    elif dataset == "SVHN" and optimizer == "ADAM":
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = None
    elif dataset == 'CIFAR100' and optimizer == "sgd":
        if dec_lr is None:
            dec_lr = [60, 120, 160]
        if gamma is None:
            gamma = 0.2
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[round(i * num_batch) for i in dec_lr],
                                                   gamma=gamma)
    elif optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    elif optimizer == "ADAM":
        print("using ADAM")
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = None
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    if privacy_engine is not None:
        privacy_engine.attach(optimizer)
    return optimizer, scheduler


def get_initial_model(model, save_path=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    if isinstance(model, str):
        try:
            architecture = eval(f"model.{model}")
        except:
            architecture = eval(f"torchvision.models.{model}")
        net = architecture().to(device)
    else:
        net = model().to(device)

    if save_path is not None:
        state = {'net': net.state_dict()}
        torch.save(state, os.path.join(save_path, f"initial_model.pt"))

    return net


def unsqueeze(architecture, parameter):
    unsqueezed = []
    net = architecture()
    reference = get_parameters(net, squeeze=False)
    for layer in reference:
        layer_shape = layer.shape
        layer_size = layer.reshape(-1).shape[0]
        unsqueezed.append(parameter[:layer_size].reshape(layer_shape))
        parameter = parameter[layer_size:]
    return unsqueezed


def add_states(state1, state2, a, b):
    return [a * i + b * j for i, j in zip(state1, state2)]


def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def get_model(model, architecture, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    state = torch.load(model)
    net = architecture()
    net.load_state_dict(state['net'])
    net.to(device)
    return net



def unnormalize(data, dataset=None, mean=None, std=None, rgb_last=False):
    if mean is None or std is None:
        normalize = [tm for tm in dataset.transform.transforms if isinstance(tm, transforms.transforms.Normalize)][0]
        mean, std = normalize.mean, normalize.std
    mean, std = mean_std_to_array(mean, std, rgb_last=rgb_last)
    if isinstance(data, torch.Tensor):
        mean, std = torch.from_numpy(mean).float(), torch.from_numpy(std).float()
    return data * std + mean


def renormalize(data, dataset=None, mean=None, std=None, rgb_last=False):
    if mean is None or std is None:
        normalize = [tm for tm in dataset.transform.transforms if isinstance(tm, transforms.transforms.Normalize)][0]
        mean, std = normalize.mean, normalize.std
    mean, std = mean_std_to_array(mean, std, rgb_last=rgb_last)
    if isinstance(data, torch.Tensor):
        mean, std = torch.from_numpy(mean).float(), torch.from_numpy(std).float()
    return (data - mean) / std


def get_save_dir(save_name):
    # ENTER SAVE DIR PATH HERE
    if os.path.exists(save_dir):
        print(save_dir)
        return os.path.join(save_dir, save_name)
    elif os.path.exists(save_dir_lab):
        return os.path.join(save_dir_lab, save_name)
    else:
        return os.path.join("models", save_name)


def get_last_ckpt(save_dir, keyword):
    saved_points = [int(model_path[len(keyword):]) for model_path in os.listdir(save_dir)
                    if keyword in model_path]
    return max(saved_points) if len(saved_points) > 0 else -1

def get_last_gen(save_dir, keyword):
    saved_gens = [int(path.split('_')[1]) for path in os.listdir(save_dir)
                    if keyword in path]
    return max(saved_gens) if len(saved_gens) > 0 else -1


def get_last_seed(save_dir, keyword):
    has_key = []
    for model_path in os.listdir(save_dir):
        if keyword in model_path:
            try: 
                has_key.append(int(model_path[len(keyword):]))
            except:
                pass
    return max(has_key) if len(has_key) > 0 else -1


def random_pos(downscale=2):
    x = np.random.normal(0, 0.5)
    while x > 1 or x < - 1:
        x = np.random.normal(0, 0.5)
    if x < 0:
        x = x / downscale / 2 + 1
    else:
        x = x / downscale / 2
    return x

