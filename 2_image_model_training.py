import time
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.pytorch_utils import check_cuda_available, get_train_test_loader, vgg16_imagenet_model, training_validation, load_model, testing


def main(image_method, until_layer, n_epochs, batch_size, use_gpu):
    start = time.time()

    train_loader, valid_loader, test_loader = get_train_test_loader(image_method,
                                                                    batch_size=batch_size,
                                                                    valid_size=0.2,
                                                                    random_seed=42,
                                                                    show=False,
                                                                    cuda=use_gpu)

    vgg16, criterion, optimizer = vgg16_imagenet_model(use_gpu, until_layer, learning_rate=0.001, verbose=False)

    vgg16, filepath = training_validation(train_loader, valid_loader, n_epochs, vgg16, criterion, optimizer, use_gpu)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    start = time.time()
    vgg16, criterion = load_model(filepath, train_on_gpu=False, verbose=False)

    total_true_labels, total_est_labels = testing(test_loader, vgg16, criterion, False)

    f_score = f1_score(total_true_labels, total_est_labels)
    precision = precision_score(total_true_labels, total_est_labels)
    recall = recall_score(total_true_labels, total_est_labels)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F score: {f_score}')

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':

    # Local import to save time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='image_method', help='Source folder of images to train.')
    parser.add_argument('--until_layer', dest='until_layer', help='Until which layer to freeze weights.')
    parser.add_argument('--n_epochs', dest='n_epochs', help='Number of epochs.')
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size')
    parser.add_argument('--use_gpu', action='store_true', dest='use_gpu', help='Using GPU')

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
    print('================================================')

    main(args.image_method, args.until_layer, int(args.n_epochs), int(args.batch_size), args.use_gpu)
