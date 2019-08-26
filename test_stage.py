import time

from utils.logger_utils import get_logger
from utils.sklearn_utils import get_performance
from utils.pytorch_utils import check_cuda_available, get_train_test_loader, testing, load_model


def main_test(image_method, filepath, time_train='', logger=None):

    if not logger:
        log_file = f'logs/{filepath.split("/")[-1].split(".")[0]}/test'
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

    get_performance(total_true_labels, total_est_labels, filepath, logger, time_train, time_test)


if __name__ == '__main__':

    image_method_ = 'RGB/undersampling/'
    filepath_ = 'data/models/1566843616_RGB_undersampling_until_30_epochs_2_batch_32.pth'
    main_test(image_method_, filepath_)

