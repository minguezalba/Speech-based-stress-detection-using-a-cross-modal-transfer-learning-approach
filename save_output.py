import time
import os
from utils.logger_utils import get_logger
from utils.sklearn_utils import get_performance
from utils.pytorch_utils import check_cuda_available, get_train_test_loader, testing, load_model
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from inspect import signature


def main_test(image_method, filepath, time_train='', logger=None):

    if not logger:
        log_dir = f'logs/{filepath.split("/")[-1].split(".")[0]}/'
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        log_file = log_dir + 'test'
        logger = get_logger(log_file)

    start = time.time()

    use_gpu = check_cuda_available()

    logger.info('\n================================================')
    logger.info(f'Testing started')
    logger.info('================================================')
    train_loader, valid_loader, test_loader = get_train_test_loader(image_method,
                                                                    valid_size=0.2,
                                                                    logger=logger,
                                                                    random_seed=42,
                                                                    show=False,
                                                                    cuda=use_gpu)

    vgg16, criterion = load_model(filepath, train_on_gpu=False, verbose=False, logger=logger)

    logger.info(f'Source folder images: {image_method}')
    logger.info(f'Model: {filepath}')
    logger.info('================================================')

    total_true_labels, total_est_labels = testing(test_loader, vgg16, criterion, False, logger)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    time_test = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    logger.info(f"\nTest elapsed time: {time_test}")
    logger.info('================================================')

    # ==================================================================================================
    dir_name = (filepath.split('/')[-1]).split('.')[0]
    dir_path = 'data/output/' + dir_name

    os.mkdir(dir_path)

    true_filename = dir_path + '/true.csv'
    np.savetxt(true_filename, total_true_labels, fmt='%i', delimiter=",")

    predicted_filename = dir_path + '/predicted.csv'
    np.savetxt(predicted_filename, total_true_labels, fmt='%i', delimiter=",")


if __name__ == '__main__':

    image_method_ = 'greyscale/oversampling/'
    filepath_ = 'data/models/1567573570_greyscale_oversampling_until_None_epochs_500_batch_32.pth'
    main_test(image_method_, filepath_)
