import glob, os, random
import numpy as np
from utils.sets_utils import split_dataset, balance, print_subset_info, create_new_directories

root_dir = 'data/images/RGB/'
input_dir = root_dir + 'imbalanced/'

stress_files = list(glob.glob(input_dir + 'stress/*.png'))
neutral_files = list(glob.glob(input_dir + 'neutral/*.png'))

labels = np.concatenate((np.ones(len(stress_files), dtype=int),
                         np.zeros(len(neutral_files), dtype=int))).reshape(-1, 1)
total_files = np.array(stress_files + neutral_files).reshape(-1, 1)

imbalance_dataset = (total_files, labels)
print_subset_info('Original set', imbalance_dataset)

method_balance = 'oversampling'
balanced_dataset = balance(imbalance_dataset, method=method_balance)
print_subset_info('Balanced set', balanced_dataset)

output_dir = root_dir + method_balance + '/'
train_subset, val_subset, test_subset = split_dataset(balanced_dataset, test_size=0.2, val_size=0.2)

for name, subset in zip(['Train', 'Validation', 'Test'], [train_subset, val_subset, test_subset]):
    print_subset_info(name, subset)

create_new_directories(output_dir, ['train', 'validation', 'test'], [train_subset, val_subset, test_subset])
