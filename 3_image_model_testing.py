import time
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.pytorch_utils import check_cuda_available, get_train_test_loader, testing, load_model


def main(image_method, filepath):

    use_gpu = check_cuda_available()

    train_loader, valid_loader, test_loader = get_train_test_loader(image_method,
                                                                    valid_size=0.2,
                                                                    random_seed=42,
                                                                    show=False,
                                                                    cuda=use_gpu)

    vgg16, criterion = load_model(filepath, train_on_gpu=False, verbose=False)

    total_true_labels, total_est_labels = testing(test_loader, vgg16, criterion, False)

    f_score = f1_score(total_true_labels, total_est_labels)
    precision = precision_score(total_true_labels, total_est_labels)
    recall = recall_score(total_true_labels, total_est_labels)

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F-Score: {f_score:.2f}')


if __name__ == '__main__':

    start = time.time()

    image_method_ = 'RGB/oversampling/'
    filepath_ = 'data/models/2019-08-04T22:20:40.699590.pth'
    main(image_method_, filepath_)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print()
    print("Test elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

