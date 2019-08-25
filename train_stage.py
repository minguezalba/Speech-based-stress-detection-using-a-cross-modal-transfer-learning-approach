import time
from test_stage import main_test
from utils.files_utils import create_dir_logs
from utils.pytorch_utils import check_cuda_available, get_train_test_loader, vgg16_imagenet_model, training_validation, load_model, testing


def main(image_method, until_layer, n_epochs, batch_size, use_gpu, do_test):
    start = time.time()

    train_loader, valid_loader, test_loader = get_train_test_loader(image_method,
                                                                    batch_size=batch_size,
                                                                    valid_size=0.2,
                                                                    random_seed=42,
                                                                    show=False,
                                                                    cuda=use_gpu)

    vgg16, criterion, optimizer = vgg16_imagenet_model(use_gpu, until_layer, learning_rate=0.001, verbose=False)

    dir_experiment = create_dir_logs(image_method, until_layer, n_epochs, batch_size)

    vgg16, model_path = training_validation(train_loader, valid_loader, n_epochs, vgg16, criterion, optimizer, use_gpu,
                                            dir_experiment)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    train_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("\nTraining elapsed time: ", train_time)

    if do_test:
        main_test(image_method, model_path, train_time)


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

    print('================================================')
    print(f'Source folder images: {args.image_method}')
    print(f'Freezing until layer: {args.until_layer}')
    print(f'Number of epochs: {args.n_epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Training on GPU: {args.use_gpu}')
    print(f'Do test after training: {args.do_test}')
    print('================================================')

    main(args.image_method, args.until_layer, int(args.n_epochs), int(args.batch_size), args.use_gpu, args.do_test)
