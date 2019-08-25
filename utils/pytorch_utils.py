import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        # print('CUDA is not available.  Training on CPU ...')
        return False
    else:
        # print('CUDA is available!  Training on GPU ...')
        return True


def plot_loss_evolution(train_losses, valid_losses, dir_experiment):
    fig = plt.figure()
    x_range = np.arange(1, len(train_losses)+1)
    plt.plot(x_range, train_losses, label='train')
    plt.plot(x_range, valid_losses, label='validation')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    # plt.show()
    filepath = dir_experiment + 'training_plot.jpg'
    plt.savefig(filepath)
    plt.close(fig)


def get_train_test_loader(image_method,
                          random_seed,
                          batch_size=BATCH_SIZE,
                          valid_size=0.1,
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

    data_dir = DIR_IMAGES + image_method
    train_dir = os.path.join(data_dir, 'train/')
    test_dir = os.path.join(data_dir, 'test/')
    val_dir = os.path.join(data_dir, 'validation/')

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

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
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
    )

    for name_set, subset in zip(['Train', 'Validation', 'Test'], [train_loader, valid_loader, test_loader]):

        labels = subset.dataset.targets

        labels = np.array(labels, dtype=int)
        print(f'{name_set}: {len(labels)} samples')
        print('\tStress: {} ({:.2f}%)'.format(np.sum(labels == 1),
                                              np.sum(labels == 1) * 100 / len(labels)))
        print('\tNeutral: {} ({:.2f}%)'.format(np.sum(labels == 0),
                                               np.sum(labels == 0) * 100 / len(labels)))

    print('================================================')
    # visualize some images
    if show:
        show_sampler = SubsetRandomSampler(list(range(9)))
        # print('show sampler: ', show_sampler.indices)
        show_loader = torch.utils.data.DataLoader(
            train_data, batch_size=9, sampler=show_sampler,
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


def modify_architecture(model, verbose):
    if verbose:
        print(f'Original VGG16 Model pre-trained on ImageNet:')
        print(model)

    n_inputs = model.classifier[6].in_features

    # add last linear layer (n_inputs -> 2 classes: neutral or stress)
    # new layers automatically have requires_grad = True
    last_layer = nn.Linear(n_inputs, len(LABELS))

    model.classifier[6] = last_layer

    if verbose:
        print('-------------------------------------------------')
        print(f'Adapted VGG16 Model pre-trained on ImageNet:')
        print(model)

    return model


def vgg16_imagenet_model(train_on_gpu, until_layer=None, learning_rate=0.001, verbose=False):

    vgg16 = models.vgg16(pretrained=True)

    if until_layer:
        for i, feature_layer in enumerate(vgg16.features[:until_layer+1]):
            for param in feature_layer.parameters():
                param.requires_grad = False

    vgg16 = modify_architecture(vgg16, verbose)

    # if GPU is available, move the model to GPU
    if train_on_gpu:
        vgg16.cuda()

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer (stochastic gradient descent) and learning rate = 0.001
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=learning_rate)

    return vgg16, criterion, optimizer


def training_validation(train_loader, valid_loader, n_epochs, vgg16, criterion, optimizer, train_on_gpu, dir_experiment):

    valid_loss_min = np.Inf  # track change in validation loss
    filepath = ''
    train_losses, valid_losses = np.zeros((n_epochs, 1)), np.zeros((n_epochs, 1))

    print(f'N epochs: {n_epochs}')
    for epoch in range(1, n_epochs + 1):
        print()
        print(f'Epoch {epoch} - {datetime.isoformat(datetime.now())}')
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
        train_losses[epoch-1] = train_loss
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_losses[epoch-1] = valid_loss

        # print training/validation statistics
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))

            filepath = save_model(dir_experiment, vgg16)
            valid_loss_min = valid_loss

    plot_loss_evolution(train_losses, valid_losses, dir_experiment)

    return vgg16, filepath


def save_model(dir_experiment, model, verbose=False):

    if verbose:
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    file_save = dir_experiment.split('/')[1]
    path_save = DIR_MODELS + file_save + EXT_MODELS
    print(f'\tSaving the model in path: {path_save}')
    torch.save(model.state_dict(), path_save)

    return path_save


def load_model(path_file, train_on_gpu, verbose):
    vgg16 = models.vgg16()
    vgg16 = modify_architecture(vgg16, verbose)

    criterion = nn.CrossEntropyLoss()

    if train_on_gpu:
        vgg16.load_state_dict(torch.load(path_file))
        vgg16.cuda()
    else:
        vgg16.load_state_dict(torch.load(path_file, map_location=lambda storage, loc: storage))

    return vgg16, criterion


def testing(test_loader, vgg16, criterion, train_on_gpu):
    print('================================================')
    print('Testing started')
    print('================================================')
    test_loss = 0.0
    n_classes = len(LABELS)
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    total_true_labels = np.array([])
    total_est_labels = np.array([])

    vgg16.eval()  # eval mode

    # iterate over test data
    for batch_i, (data, target) in enumerate(test_loader):

        print('{}/{}'.format(batch_i, len(test_loader)))
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
        # _ would be the max value of the output vector, that is, the max value of probability
        # pred is the index of the max value of the probability, in this case would be the predicted class, 0 or 1,
        # along columns
        _, pred = torch.max(output, 1)
        pred_np = np.squeeze(pred.numpy()) if not train_on_gpu else np.squeeze(pred.cpu().numpy())
        total_est_labels = np.concatenate([total_est_labels, pred_np]) if total_est_labels.size else pred_np

        # compare predictions to true label
        true_labels_tensor = target.data.view_as(pred)
        true_labels_np = np.squeeze(true_labels_tensor.numpy()) if not train_on_gpu else np.squeeze(true_labels_tensor.cpu().numpy())
        total_true_labels = np.concatenate([total_true_labels, true_labels_np]) if total_true_labels.size else true_labels_np

        correct_tensor = pred.eq(true_labels_tensor)
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

    return total_true_labels, total_est_labels
