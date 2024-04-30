import numpy as np
import pandas as pd
import os
import pwd
import scipy
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as torch_utils
import torchvision.utils as torch_utils2
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.titlepad': 5,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1,
    'axes.labelpad': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.pad': 4,
    'ytick.major.pad': 4,
    'legend.title_fontsize': 14,
    'legend.frameon': False,
    'lines.markersize': 1.3,
    'legend.markerscale': 2
})

class FairFace(datasets.VisionDataset):
    def __init__(self, root=f'/scratch/ssd004/scratch/{pwd.getpwuid(os.getuid())[0]}', train=True, valid=False, transform=None, target_transform=None, **kwargs):
        super(FairFace, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        # check if annotations are ready to go
        # if not os.path.isfile('./data/fairface/annotations-train.csv'):
        # self.prep_csvs()

        if train: # train
            self.csv = './data/fairface/annotations-train.csv'
        elif valid: # valid
            self.csv = './data/fairface/annotations-valid.csv'
        else: # test
            self.csv = './data/fairface/annotations-test.csv'
        self.imdir = root        
        df = pd.read_csv(self.csv)
        self.sensitive_one = df['race']
        self.sensitive_two = df['age']
        self.label = df['gender']
        self.image_names = df['file']
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
        sens_one = self.sensitive_one[index]
        sens_two = self.sensitive_two[index]
        # Convert to int. maps in order of population size in train set 
        gender_map = {'Male': 0,
                     'Female': 1}
        youngs = ['0-2', '3-9', '10-19', '20-29']
        olds = ['30-39', '40-49', '50-59', '60-69', 'more than 70']
        race_map = {'White': 0,
                    'Latino_Hispanic': 1,
                    'Indian': 2,
                    'East Asian': 3,
                    'Black': 4,
                    'Southeast Asian': 5,
                    'Middle Eastern': 6}
        lab = gender_map[lab]
        sens_one = race_map[sens_one]
        sens_two = 0 if sens_two in youngs else 1
        sens = [sens_one, sens_two]
        return img, sens, lab, index

    def __len__(self):
        return len(self.image_names)
    
    def prep_csvs(self):
        # prep annotations file
        train_ano = pd.read_csv(os.path.join(self.root, 'fairface_label_train.csv'))
        train_ano = train_ano[10954:]
        train_ano.to_csv('./data/fairface/annotations-train.csv', index=None)

        test_ano = pd.read_csv(os.path.join(self.root, 'fairface_label_train.csv'))
        test_ano = test_ano[:10954]
        test_ano = test_ano.drop(columns=['file']) # drop and add new file names
        fnames = [f"test/{i}.jpg" for i in range(1, 10955)]
        test_ano['file'] = fnames
        test_ano.to_csv('./data/fairface/annotations-test.csv', index=None)
        
        valid_ano = pd.read_csv(os.path.join(self.root, 'fairface_label_val.csv'))
        valid_ano.to_csv('./data/fairface/annotations-valid.csv', index=None)



if __name__ == '__main__':
    # Binary AGE x RACE to predict GENDER
    # dset = FairFace()
    # folder_dir = f'/scratch/ssd004/scratch/{pwd.getpwuid(os.getuid())[0]}/'
    # train_ano = pd.read_csv(os.path.join(folder_dir, 'fairface_label_train.csv'))
    train_ano = pd.read_csv('./data/fairface/annotations-train.csv')
    # print(train_ano.columns)

    # Train distr stats
    # for frame in [train_ano, valid_ano]:
    #     print(f"Length of frame: {len(frame)}")
    #     print(frame.value_counts('gender')/len(frame)*100)
    #     print(frame.value_counts('race')/len(frame)*100)
    #     print(frame.value_counts('age')/len(frame)*100)
    #     print("\n\n")


    # # intersectional stats (predicting age, protecting race and gender)
    genders_data = train_ano.value_counts('gender')
    genders = genders_data.index.tolist()
    genders_data = genders_data / np.sum(genders_data)
    print(genders_data)
    fig, ax = plt.subplots()
    plt.figure(figsize=(4,1))
    plt.bar(x=genders, height=genders_data)
    plt.title('Gender Balance')
    plt.ylabel('Representation')
    # plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    plt.savefig('./fairface_genders.png', dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(genders)

    races_data = train_ano.value_counts('race')
    races = races_data.index.tolist()
    races_data = races_data / np.sum(races_data)
    fig, ax = plt.subplots()
    plt.figure(figsize=(10,6))
    plt.bar(x=races, height=races_data)
    plt.title('Race Balance')
    plt.ylabel('Representation')
    plt.xticks(ticks=range(7), labels=races, rotation=45, ha="right", rotation_mode="anchor")
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.savefig('./fairface_races.png', dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(races)

    youngs = ['0-2', '3-9', '10-19', '20-29']
    olds = ['30-39', '40-49', '50-59', '60-69', 'more than 70']
    ages_data = train_ano.value_counts('age')
    young_data = np.sum(ages_data[youngs]) / np.sum(ages_data)
    old_data = np.sum(ages_data[olds]) / np.sum(ages_data)
    fig, ax = plt.subplots()
    plt.figure(figsize=(4,1))
    plt.bar(x=["≤29", ">30"], height=[young_data, old_data])
    plt.title('Age Balance (Binarized)')
    plt.ylabel('Representation')
    # plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    plt.savefig('./fairface_ages.png', dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"<30: \t{young_data} \n>=30\t{old_data}")

    ages = ages_data.index.tolist()
    print(ages)

    gender_map = {0: 'Male',
                1:'Female'}
    
    race_map = {0: 'White',
                1: 'Latino/Hispanic',
                2: 'Indian',
                3: 'East Asian',
                4: 'Black',
                5: 'Southeast Asian',
                6: 'Middle Eastern'}
    
    og_gender_map = {'Male': 0,
                     'Female': 1}
    og_youngs = ['0-2', '3-9', '10-19', '20-29']
    og_olds = ['30-39', '40-49', '50-59', '60-69', 'more than 70']
    og_race_map = {'White': 0,
                'Latino_Hispanic': 1,
                'Indian': 2,
                'East Asian': 3,
                'Black': 4,
                'Southeast Asian': 5,
                'Middle Eastern': 6}
    
    num_cats = len(genders) * len(races) * 2
    fair_ideal = np.array([1/num_cats] * num_cats)
    fair_counts = np.floor(fair_ideal * train_ano.shape[0]) 

    # get data and encode
    sens1 = np.array(train_ano['race'])
    sens2 = np.array(train_ano['age'])
    labels = np.array(train_ano['gender'])
    for i in range(len(sens1)):
        sens1[i] = og_race_map[sens1[i]]
        sens2[i] = 0 if sens2[i] in og_youngs else 1
        labels[i] = og_gender_map[labels[i]]

    print(f"sens2 {np.sum(sens2)}, lab {np.sum(labels)}")
    
    n_sens = [len(races), 2]
    sens_list = [list(range(n_)) for n_ in n_sens]
    s1, s2 = np.meshgrid( *sens_list)
    s1 = s1.flatten()
    s2 = s2.flatten()
    cat_idxs = []
    headers = []
    frames = np.zeros((2, 7, 2))
    for gender in [0, 1]:
        for race in range(7):
            for age in range(2):
                freq = len(np.intersect1d(np.where(labels==gender), np.intersect1d(np.where(sens1==race), np.where(sens2==age))))
                frames[gender, race, age] = freq

    normed = frames/np.sum(frames)
    print(normed.shape)

    fig, (ax1, ax2) = plt.subplots(nrows=2)  
    # plt.figure(figsize=(10,6))
    # add a big axis, hide frame
    big = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    ax1.imshow(normed[0].T, interpolation="nearest", origin="upper")
    ax1.set_xticks(ticks=[])
    ax1.set_title(gender_map[0])
    ax1.set_yticks(ticks=list(range(2)), labels=["≤29", ">30"])
    # ax1.set_ylabel('Age')
    im = ax2.imshow(normed[1].T, interpolation="nearest", origin="upper")
    ax2.set_xticks(ticks=range(7), labels=races)
    ax2.set_title(gender_map[1])
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax2.set_yticks(ticks=list(range(2)), labels=["≤29", ">30"])
    plt.ylabel('Age')
    ax2.set_xlabel('Race')
    cbar = fig.colorbar(im, ax=[ax1, ax2])
    cbar.ax.set_ylabel('Frequency')
    plt.savefig(f"./fairface_strata.png", dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()

    # actual = []
    # for frame in [train_ano]:
    #     for gender in genders:
    #         for age in ages:
    #             subframe = frame.where(frame['gender']==gender)
    #             subframe = subframe.where(subframe['age']==age)
    #             row = subframe.value_counts('race')/len(frame)
    #             actual.append(row)
    #     # print("\n\n")

    # # define fairness ideal:
    # ideal = np.asarray([1 / (len(races)*len(genders))] * (len(races) * len(genders) * len(ages)))
    # # print(ideal)
    # print(f"Number categories to balance: {len(ideal)}") # 126. Need big batches if we have a hope of conducting STAR (256 at least)
    # # train set balance: Male, races, then Female, races. Order doesnt matter given out fairness ideal
    # actual = np.asarray(actual).flatten()
    # # print(actual)

    # # initial KL-divergence
    # kl_div = scipy.stats.entropy(actual, ideal)
    # print(f"KL Divergence: {kl_div}")
