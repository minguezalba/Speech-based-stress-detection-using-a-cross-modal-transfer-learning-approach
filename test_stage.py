import time

import numpy as np
from utils.sklearn_utils import get_performance
from utils.pytorch_utils import check_cuda_available, get_train_test_loader, testing, load_model


def main_test(image_method, filepath, time_train=''):

    start = time.time()

    use_gpu = check_cuda_available()

    print('\n================================================')
    print(f'Testing started')
    print('================================================')
    train_loader, valid_loader, test_loader = get_train_test_loader(image_method,
                                                                    valid_size=0.2,
                                                                    random_seed=42,
                                                                    show=False,
                                                                    cuda=use_gpu)

    vgg16, criterion = load_model(filepath, train_on_gpu=False, verbose=False)

    print(f'Source folder images: {image_method}')
    print(f'Model: {filepath}')
    print('================================================')

    total_true_labels, total_est_labels = testing(test_loader, vgg16, criterion, False)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    time_test = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("\nTest elapsed time: ", time_test)
    print('================================================')

    get_performance(total_true_labels, total_est_labels, filepath, time_train, time_test)


if __name__ == '__main__':

    image_method_ = 'greyscale/oversampling/'
    filepath_ = 'data/models/2019-08-21T16:15:08.870886.pth'
    main_test(image_method_, filepath_)

