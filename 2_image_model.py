from utils.pytorch_utils import check_cuda_available, get_train_test_loader


def main(image_method):

    cuda = check_cuda_available()

    train_loader, test_loader = get_train_test_loader(image_method,
                                                      test_size=0.2,
                                                      random_seed=42,
                                                      show=True,
                                                      cuda=cuda)


    # vgg16_model()
    #
    # training()
    #
    # testing()


if __name__ == '__main__':
    image_method_ = 'RGB'
    main(image_method_)
