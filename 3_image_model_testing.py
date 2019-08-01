import time

from utils.pytorch_utils import check_cuda_available, get_train_test_loader, testing, load_model


def main(image_method, filepath):

    use_gpu = check_cuda_available()

    train_loader, valid_loader, test_loader = get_train_test_loader(image_method,
                                                                    valid_size=0.2,
                                                                    random_seed=42,
                                                                    show=False,
                                                                    cuda=use_gpu)

    vgg16, criterion = load_model(filepath, False)

    testing(test_loader, vgg16, criterion, False)


if __name__ == '__main__':

    start = time.time()

    image_method_ = 'RGB'
    filepath_ = 'data/models/2019-08-01.pth'
    main(image_method_, filepath_)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

