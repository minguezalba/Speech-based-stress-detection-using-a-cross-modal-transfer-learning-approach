import time

from utils.pytorch_utils import check_cuda_available, get_train_test_loader, vgg16_imagenet_model, training_validation


def main(image_method, until_layer, n_epochs, batch_size, use_gpu):

    train_loader, valid_loader, test_loader = get_train_test_loader(image_method,
                                                                    batch_size=batch_size,
                                                                    valid_size=0.2,
                                                                    random_seed=42,
                                                                    show=False,
                                                                    cuda=use_gpu)

    vgg16, criterion, optimizer = vgg16_imagenet_model(use_gpu, until_layer, learning_rate=0.001, verbose=False)

    vgg16, filepath = training_validation(train_loader, valid_loader, n_epochs, vgg16, criterion, optimizer, use_gpu)


if __name__ == '__main__':

    start = time.time()

    image_method_ = 'RGB/balanced/'
    until_layer_ = 2   # Layers from 0 to 30
    n_epochs_ = 100
    batch_size_ = 20
    use_gpu = check_cuda_available()
    # use_gpu = False

    print('================================================')
    print(f'Source folder images: {image_method_}')
    print(f'Freezing until layer: {until_layer_}')
    print(f'Number of epochs: {n_epochs_}')
    print(f'Batch size: {batch_size_}')
    print(f'Training on GPU: {use_gpu}')
    print('================================================')

    main(image_method_, until_layer_, n_epochs_, batch_size_, use_gpu)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
