import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data.sampler import SubsetRandomSampler
# from utils import plot_images
from torchvision import datasets, transforms, models

from utils.image_utils import DIR_IMAGES

LABELS = [
    'neutral',
    'stress'
]

BATCH_SIZE = 40
NUM_WORKERS = 4

DIR_MODELS = 'data/models/'
EXT_MODELS = '.pth'


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
                          valid_size=0.1,
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

    print('Generating dataloaders...')

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

    # split between training and test
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(np.floor(test_size * num_samples))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    # split between training and validation
    num_train_samples = len(train_idx)
    split = int(np.floor(valid_size * num_train_samples))

    train_idx, valid_idx = train_idx[split:], train_idx[:split]

    print('Dataloaders info:')
    print('# Train samples: {}'.format(len(train_idx)))
    print(train_idx[:10])
    print('# Validation samples: {}'.format(len(valid_idx)))
    print(valid_idx[:10])
    print('# Test samples: {}'.format(len(test_idx)))
    print(test_idx[:10])

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show:
        show_sampler = SubsetRandomSampler(list(range(9)))
        # print('show sampler: ', show_sampler.indices)
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

    return train_loader, valid_loader, test_loader


def vgg16_imagenet_model(train_on_gpu, learning_rate=0.001, verbose=False):

    print('Defining and adapting the pre-trained model...')

    vgg16 = models.vgg16(pretrained=True)
    if verbose:
        print(f'Original VGG16 Model pre-trained on ImageNet:')
        print(vgg16)

    # Freeze training for all "features" layers
    for param in vgg16.features.parameters():
        param.requires_grad = False

    n_inputs = vgg16.classifier[6].in_features

    # add last linear layer (n_inputs -> 2 classes: neutral or stress)
    # new layers automatically have requires_grad = True
    last_layer = nn.Linear(n_inputs, len(LABELS))

    vgg16.classifier[6] = last_layer

    # if GPU is available, move the model to GPU
    if train_on_gpu:
        vgg16.cuda()

    # check to see that your last layer produces the expected number of outputs
    # print(vgg16.classifier[6].out_features)

    if verbose:
        print('-------------------------------------------------')
        print(f'Adapted VGG16 Model pre-trained on ImageNet:')
        print(vgg16)

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer (stochastic gradient descent) and learning rate = 0.001
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=learning_rate)

    return vgg16, criterion, optimizer


def training_validation(train_loader, valid_loader, n_epochs, vgg16, criterion, optimizer, train_on_gpu):

    valid_loss_min = np.Inf  # track change in validation loss
    filepath = ''

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # =====================
        # train the model #
        # =====================
        vgg16.train()
        for batch_i, (data, target) in enumerate(train_loader):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = vgg16(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        # =====================
        # validate the model #
        # =====================
        vgg16.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = vgg16(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        # print training/validation statistics
        print()
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            filepath = save_model(vgg16)
            valid_loss_min = valid_loss

    return vgg16, filepath


# def training(train_loader, n_epochs, vgg16, criterion, optimizer, train_on_gpu):
#
#     print('Training the model...')
#
#     for epoch in range(1, n_epochs + 1):
#
#         # keep track of training and validation loss
#         train_loss = 0.0
#
#         ###################
#         # train the model #
#         ###################
#         # model by default is set to train
#         for batch_i, (data, target) in enumerate(train_loader):
#             # move tensors to GPU if CUDA is available
#             if train_on_gpu:
#                 data, target = data.cuda(), target.cuda()
#             # clear the gradients of all optimized variables
#             optimizer.zero_grad()
#             # forward pass: compute predicted outputs by passing inputs to the model
#             output = vgg16(data)
#             # calculate the batch loss
#             loss = criterion(output, target)
#             # backward pass: compute gradient of the loss with respect to model parameters
#             loss.backward()
#             # perform a single optimization step (parameter update)
#             optimizer.step()
#             # update training loss
#             train_loss += loss.item()
#
#             if batch_i % 20 == 19:  # print training loss every specified number of mini-batches
#                 print('Epoch %d, Batch %d loss: %.16f' %
#                       (epoch, batch_i + 1, train_loss / 20))
#                 train_loss = 0.0
#
#     return vgg16, optimizer


def save_model(model, verbose=False):

    print('Saving the model in path:')

    if verbose:
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    path_save = DIR_MODELS + datetime.isoformat(datetime.now()) + EXT_MODELS
    print(path_save)
    torch.save(model.state_dict(), path_save)

    return path_save


def load_model(path_file):
    vgg16 = models.vgg16()
    vgg16.load_state_dict(torch.load(path_file))
    vgg16.eval()

    return vgg16


def testing(test_loader, vgg16, criterion, train_on_gpu):
    test_loss = 0.0
    n_classes = len(LABELS)
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))

    vgg16.eval()  # eval mode

    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model
        output = vgg16(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update  test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(n_classes):
        if class_total[i] > 0:
            print('Test Accuracy of %2s: %2d%% (%2d/%2d)' % (LABELS[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %2s: N/A (no training examples)' % (LABELS[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
