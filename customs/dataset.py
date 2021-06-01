from torchvision import datasets, transforms
import torch

def transform(opt):
    if opt.transform == 'default':
        transform = transform=transforms.ToTensor()
    return transform

def mnist(opt, train=True):
    if train:
        dataset = datasets.MNIST(root='D:\Code\PythonCode\general_pytorch_templet\dataset', train=train, transform=transform(opt), download=True)
        validation_size = int(opt.val_rate * len(dataset))
        train_size = len(dataset) - validation_size
        return torch.utils.data.random_split(dataset, [train_size, validation_size])
    else:
        return datasets.MNIST(root='D:\Code\PythonCode\general_pytorch_templet\dataset', train=train, transform=transform(opt), download=True)

def load_dataset(opt, train=True):
    if opt.dataset == 'mnist':
        return mnist(opt, train)
    else:
        print('err: there is no dataset')