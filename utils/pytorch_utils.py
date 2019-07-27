
import torch
import numpy as np
import matplotlib.pyplot as plt

# from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils.image_utils import DIR_IMAGES

LABELS = [
    'neutral',
    'stress'
]

BATCH_SIZE = 20
NUM_WORKERS = 4


def check_cuda_available():
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
        return False
    else:
        print('CUDA is available!  Training on GPU ...')
        return True


def get_train_test_loader(image_method,
                          random_seed,
                          batch_size=BATCH_SIZE,
                          test_size=0.1,
                          shuffle=True,
                          show=False,
                          num_workers=NUM_WORKERS,
                          cuda=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - test_loader: test set iterator.
    """

    # Check and transform parameters
    pin_memory = True if cuda else False

    data_dir = DIR_IMAGES + image_method + '/'

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((test_size >= 0) and (test_size <= 1)), error_msg

    # VGG16 pytorch model normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # define transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(np.floor(test_size * num_samples))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    print(f'Train indexes: {train_idx[:10]}')
    print(f'Test indexes: {test_idx[:10]}')

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show:
        show_sampler = SubsetRandomSampler(list(range(9)))
        print('show sampler: ', show_sampler.indices)
        show_loader = torch.utils.data.DataLoader(
            dataset, batch_size=9, sampler=show_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # obtain one batch of training images
        data_iter = iter(show_loader)
        images, labels = next(data_iter)
        images = images.numpy()  # convert images to numpy for display

        # plot the images in the batch, along with the corresponding labels
        fig, axes = plt.subplots(3, 3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(np.transpose(images[i], (1, 2, 0)))

            cls_true_name = LABELS[labels[i]]
            x_label = "{0} ({1})".format(cls_true_name, labels[i])

            ax.set_xlabel(x_label)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()

    return train_loader, test_loader
