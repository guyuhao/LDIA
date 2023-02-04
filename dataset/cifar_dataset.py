import torchvision
from torchvision import transforms

from dataset.utils import generate_loaders, \
    get_dataset_labels, client_split, auxiliary_test_partition

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def load_data(name, args,
              partition=False, probability_list=None, quantity=None):
    """
    load train or test data of CIFAR-10 from local file

    :param name: 'train' or 'test'
    :param args: configuration
    :param partition: whether to partition dataset, True for train dataset
    :param probability_list: preset Dirichlet distribution
    :param quantity: the value of n for #C=n
    :return: dataset
    """
    if probability_list is None:
        probability_list = []
    dataset_path = '../../data'
    dataset = torchvision.datasets.CIFAR10(root=dataset_path,
                                           train=True if name == 'train' else False,
                                           download=True,
                                           transform=transform)
    y = get_dataset_labels(dataset, args['dataset'])

    if partition:
        return client_split(
            dataset=dataset, x=None, y=y, name=name, args=args,
            probability_list=probability_list, quantity=quantity)
    else:
        return dataset


def get_dataloader(args, shadow_non_iid=None, quantity=None):
    """
    generate loader of CIFAR-10 dataset

    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :param quantity: the value of n for #C=n data distribution setting
    :return: tuple contains:
        (1) train_loaders: loaders of users' datasets;
        (2) test_loader: loader of test dataset;
        (3) auxiliary_loader: loader of auxiliary dataset
        (4) probe_loader: loader of probe dataset, used by Updates-Leak
    """
    train_loaders = get_train_loaders(args, quantity=quantity)
    test_loader, auxiliary_loader, probe_loader = \
        get_test_auxiliary_loaders(args, shadow_non_iid=shadow_non_iid)
    return train_loaders, test_loader, auxiliary_loader, probe_loader


def get_train_loaders(args, quantity=None):
    """
    generate loaders of users' datasets

    :param args: configuration
    :param quantity: the value of n for #C=n data distribution setting
    :return: loaders of users' datasets
    """
    batch_size = args['target_batch_size']
    train_datasets = load_data('train', args, partition=True, quantity=quantity)
    if isinstance(train_datasets, tuple):
        train_datasets, probability_list = train_datasets
    train_loaders = generate_loaders(train_datasets, batch_size)
    return train_loaders


def get_test_auxiliary_loaders(args, shadow_non_iid=None):
    """
    generate loaders of test dataset, auxiliary dataset and probe dataset from the original test data

    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :return: tuple contains:
        (1) test_loader: loader of test dataset;
        (2) auxiliary_loader: loader of auxiliary dataset
        (3) probe_loader: loader of probe dataset, used by Updates-Leak
    """
    batch_size = args['target_batch_size']
    test_dataset = load_data('test', args, partition=False)
    auxiliary_dataset, probe_dataset, test_dataset = \
        auxiliary_test_partition(test_dataset, args, shadow_non_iid=shadow_non_iid)
    auxiliary_loader = generate_loaders([auxiliary_dataset], batch_size)[0]
    probe_loader = generate_loaders([probe_dataset], batch_size)[0]
    test_loader = generate_loaders([test_dataset], batch_size)[0]
    return test_loader, auxiliary_loader, probe_loader


