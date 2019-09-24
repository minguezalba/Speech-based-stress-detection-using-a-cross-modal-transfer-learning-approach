import time
from test_stage import main_test
from utils.files_utils import create_dir_logs
from utils.logger_utils import get_logger
from utils.pytorch_utils import check_cuda_available, get_train_test_loader, vgg16_imagenet_model, training_validation, load_model, testing


def main(image_method, until_layer, n_epochs, batch_size, use_gpu, do_test, logger):
    start = time.time()

    train_loader, valid_loader, test_loader = get_train_test_loader(image_method,
                                                                    batch_size=batch_size,
                                                                    logger=logger,
                                                                    valid_size=0.2,
                                                                    random_seed=42,
                                                                    show=False,
                                                                    cuda=use_gpu)

    vgg16, criterion, optimizer = vgg16_imagenet_model(use_gpu, logger, until_layer, learning_rate=0.01, verbose=False)

    vgg16, model_path = training_validation(train_loader, valid_loader, n_epochs, vgg16, criterion, optimizer, use_gpu,
                                            dir_experiment, logger)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    train_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    logger.info(f"\nTraining elapsed time: {train_time}")
    logger.info('================================================')

    if do_test:
        main_test(image_method, model_path, train_time, logger)


if __name__ == '__main__':

    # Local import to save time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='image_method', help='Source folder of images to train.')
    parser.add_argument('--until_layer', dest='until_layer', help='Until which layer to freeze weights.')
    parser.add_argument('--n_epochs', dest='n_epochs', help='Number of epochs.')
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size')
    parser.add_argument('--use_gpu', action='store_true', dest='use_gpu', help='Using GPU')
    parser.add_argument('--do_test', action='store_true', dest='do_test', help='Do test after training')

    args = parser.parse_args()
    if not args.use_gpu:
        use_gpu = check_cuda_available()
        args.use_gpu = use_gpu

    args.until_layer = int(args.until_layer)
    if args.until_layer == -1:
        args.until_layer = None

    dir_experiment = create_dir_logs(args.image_method, args.until_layer, args.n_epochs, args.batch_size)

    if args.do_test:
        log_file = f'{dir_experiment}training_test'
    else:
        log_file = f'{dir_experiment}training'

    logger_ = get_logger(log_file)
    
    logger_.info('================================================')
    logger_.info(f'Source folder images: {args.image_method}')
    logger_.info(f'Freezing until layer: {args.until_layer}')
    logger_.info(f'Number of epochs: {args.n_epochs}')
    logger_.info(f'Batch size: {args.batch_size}')
    logger_.info(f'Training on GPU: {args.use_gpu}')
    logger_.info(f'Do test after training: {args.do_test}')
    logger_.info('================================================')

    main(args.image_method, args.until_layer, int(args.n_epochs), int(args.batch_size), args.use_gpu, args.do_test,
         logger_)

    for handler in logger_.handlers:
        handler.close()
        logger_.removeFilter(handler)


