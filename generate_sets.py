import glob, os, random
import numpy as np
from utils.sets_utils import split_dataset, balance, print_subset_info, create_new_directories

root_dir = 'data/images/greyscale/'
# root_dir = 'data/images/RGB/'
input_dir = root_dir + 'imbalanced/'

stress_files = list(glob.glob(input_dir + 'stress/*.png'))
neutral_files = list(glob.glob(input_dir + 'neutral/*.png'))

labels = np.concatenate((np.ones(len(stress_files), dtype=int),
                         np.zeros(len(neutral_files), dtype=int))).reshape(-1, 1)
total_files = np.array(stress_files + neutral_files).reshape(-1, 1)

imbalance_dataset = (total_files, labels)
print_subset_info('\nOriginal set', imbalance_dataset)

# method_balance = 'undersampling'
method_balance = 'oversampling'

output_dir = root_dir + method_balance + '/'
train, val, test = split_dataset(imbalance_dataset, test_size=0.2, val_size=0.2)

print('\nBefore balancing:')
for name, subset in zip(['Train', 'Validation', 'Test'], [train, val, test]):
    print_subset_info(name, subset)

balanced_train = balance(train, method=method_balance)
balanced_val = balance(val, method=method_balance)

print('\nAfter balancing:')
for name, subset in zip(['Train', 'Validation', 'Test'], [balanced_train, balanced_val, test]):
    print_subset_info(name, subset)

create_new_directories(output_dir, ['train', 'validation', 'test'], [balanced_train, balanced_val, test])
