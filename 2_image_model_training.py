import time

from utils.pytorch_utils import check_cuda_available, get_train_test_loader, vgg16_imagenet_model, training_validation


def main(image_method):

    use_gpu = check_cuda_available()

    train_loader, valid_loader, test_loader = get_train_test_loader(image_method,
                                                                    valid_size=0.2,
                                                                    test_size=0.2,
                                                                    random_seed=42,
                                                                    show=False,
                                                                    cuda=use_gpu)

    vgg16, criterion, optimizer = vgg16_imagenet_model(use_gpu, learning_rate=0.001)

    n_epochs = 2
    vgg16, filepath = training_validation(train_loader, valid_loader, n_epochs, vgg16, criterion, optimizer, use_gpu)


if __name__ == '__main__':

    start = time.time()

    image_method_ = 'RGB'
    main(image_method_)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
