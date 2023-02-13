from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist(args):
    mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

def get_cifar(args):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.201]

    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=mean, std=std)])

    train_data = datasets.CIFAR10(".", transform=transform, train=True, download=True)
    test_data = datasets.CIFAR10(".", transform=transform, train=False, download=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader

def get_data(args):
    if args.data == 'mnist':
        input_shape=[1,28,28]
        output_dim=10
        return (*get_mnist(args), input_shape, output_dim)
    if args.data == 'cifar10':
        print("Loading CIFAR10")
        input_shape=[3,32,32]
        output_dim=10
        return (*get_cifar(args), input_shape, output_dim)
    else:
        return NotImplementedError("Data {} unknown".format(args.data))
