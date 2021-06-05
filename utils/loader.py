import torch,torchvision,os
import torchvision.transforms as transforms
import numpy as np

def get_loader(dset,batch_size,test_size=None,test_batch_size=500):
    if dset == 'MNIST':
        transf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        # MNIST Dataset (Images and Labels)
        train_dataset = torchvision.datasets.MNIST(root =os.path.abspath('../data'),train=True,transform=transf, download=True)
        test_dataset = torchvision.datasets.MNIST(root =os.path.abspath('../data'),train=False,transform=transf, download=False)
        if test_size is not None and test_size < 10000:
            np.random.seed(42)
            sampled_index=np.random.choice(10000,test_size)
            test_dataset.data = torch.tensor(np.array(test_dataset.data)[sampled_index])
            test_dataset.targets = torch.tensor(np.array(test_dataset.targets)[sampled_index])


        # Dataset Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=test_batch_size, shuffle = False)

    elif dset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False,download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False)
    
    elif dset == 'CIFAR100':
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR100(root='../data', train=False,download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False)
        
    
    else:
        print('unrecogonized dataset')
        exit(1)

    return train_loader,test_loader
