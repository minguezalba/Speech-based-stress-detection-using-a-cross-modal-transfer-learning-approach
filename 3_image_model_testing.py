from utils.pytorch_utils import check_cuda_available, get_train_test_loader, vgg16_imagenet_model, training, testing, \
    save_model, load_model


def main(image_method, filepath):

    use_gpu = check_cuda_available()

    vgg16 = load_model()

    testing(test_loader, vgg16, criterion, use_gpu)


if __name__ == '__main__':
    image_method_ = 'RGB'
    main(image_method_)
