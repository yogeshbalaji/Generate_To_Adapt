from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from logger import Logger
import numpy as np
import sys
import trainer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to source dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Number of filters to use in the generator network')
    parser.add_argument('--ndf', type=int, default=64, help='Number of filters to use in the discriminator network')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
    parser.add_argument('--gpu', type=int, default=1, help='GPU to use, -1 for CPU training')
    parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
    parser.add_argument('--method', default='GTA', help='Method to train| GTA, sourceonly')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--adv_weight', type=float, default = 0.1, help='weight for adv loss')
    parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate decay, default=0.0002')
    parser.add_argument('--alpha', type=float, default = 0.3, help='multiplicative factor for target adv. loss')
    parser.add_argument('--classes', type=int, default = 10, help='Total number of classes to ')

    opt = parser.parse_args()
    print(opt)

    # Creating log directory
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'visualization'))
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'models'))
    except OSError:
        pass


    # Setting random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.gpu>=0:
        torch.cuda.manual_seed_all(opt.manualSeed)

    # GPU/CPU flags
    cudnn.benchmark = True
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --gpu [gpu id]")
    if opt.gpu>=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    # Creating data loaders
    mean = np.array([0.44, 0.44, 0.44])
    std = np.array([0.19, 0.19, 0.19])

    source_train_root = os.path.join(opt.dataroot, 'svhn/trainset')
    source_val_root = os.path.join(opt.dataroot, 'svhn/testset')
    target_root = os.path.join(opt.dataroot, 'mnist/trainset')
    
    transform_source = transforms.Compose([transforms.Resize(opt.imageSize), transforms.ToTensor(), transforms.Normalize(mean,std)])
    transform_target = transforms.Compose([transforms.Resize(opt.imageSize), transforms.ToTensor(), transforms.Normalize(mean,std)])

    source_train = dset.ImageFolder(root=source_train_root, transform=transform_source)
    source_val = dset.ImageFolder(root=source_val_root, transform=transform_source)
    target_train = dset.ImageFolder(root=target_root, transform=transform_target)


    #validation_split = .2
    
    # Creating data indices for training and validation splits:
    #dataset_size = len(source_train)
    #indices = list(range(dataset_size))
    #split = int(np.floor(validation_split * dataset_size))
    #train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    #train_sampler = SubsetRandomSampler(train_indices)
    #valid_sampler = SubsetRandomSampler(val_indices)

    source_trainloader = torch.utils.data.DataLoader(source_train, batch_size=opt.batchSize, shuffle=True, num_workers=2, drop_last=True)
    source_valloader = torch.utils.data.DataLoader(source_val, batch_size=opt.batchSize, shuffle=False, num_workers=2, drop_last=False)
    targetloader = torch.utils.data.DataLoader(target_train, batch_size=opt.batchSize, shuffle=True, num_workers=2, drop_last=True)

    nclasses = len(source_train.classes)
    print(nclasses)
    
    # Training
    if opt.method == 'GTA':
        GTA_trainer = trainer.GTA(opt, nclasses, mean, std, source_trainloader, source_valloader, targetloader)
        GTA_trainer.train()
    elif opt.method == 'sourceonly':
        sourceonly_trainer = trainer.Sourceonly(opt, nclasses, source_trainloader, source_valloader)
        sourceonly_trainer.train()
    else:
        raise ValueError('method argument should be GTA or sourceonly')

if __name__ == '__main__':
    main()

