import numpy as np
import sys, os
from shutil import copyfile
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def balance(imbalance_dataset, method):
    X_resampled, y_resampled = [], []

    if method == 'oversampling':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(*imbalance_dataset)

    elif method == 'undersampling':
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(*imbalance_dataset)

    else:
        print('Method error.')
        sys.exit()

    return X_resampled, y_resampled


def split_dataset(dataset, test_size, val_size):
    
    files, labels = dataset
    
    n_files = len(files)
    indexes = np.arange(n_files)
    np.random.seed(42)
    np.random.shuffle(indexes)

    split_test = int(test_size * n_files)
    train_index = indexes[split_test:]
    test_index = indexes[:split_test]

    split_test = int(val_size * len(train_index))
    train_index = train_index[split_test:]
    val_index = train_index[:split_test]
    
    train_files, train_labels = files[train_index], labels[train_index]
    val_files, val_labels = files[val_index], labels[val_index]
    test_files, test_labels = files[test_index], labels[test_index]

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


def print_subset_info(name, subset):
    print(name)
    labels = subset[1]
    print('\tStress: {} ({:.2f}%)'.format(np.sum(labels == 1), np.sum(labels == 1)*100/len(labels)))
    print('\tNeutral: {} ({:.2f}%)'.format(np.sum(labels == 0), np.sum(labels == 0)*100/len(labels)))


def create_new_directories(output_dir, list_names, list_subsets):
    for name, subset in zip(list_names, list_subsets):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        if not os.path.isdir(output_dir + name):
            os.mkdir(output_dir + name)
            os.mkdir(output_dir + name + '/stress')
            os.mkdir(output_dir + name + '/neutral')

        files = np.squeeze(subset[0])
        labels = np.squeeze(subset[1])

        for file, label in zip(files, labels):
            filename = file.split('/')[-1]
            if label == 1:
                new_path = output_dir + name + '/stress/' + filename
            else:
                new_path = output_dir + name + '/neutral/' + filename

            copyfile(file, new_path)
