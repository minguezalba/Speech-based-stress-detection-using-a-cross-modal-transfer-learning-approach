import time
import os
import numpy as np
import glob
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from inspect import signature

VARIABLES = {
    'image_method': ['RGB', 'greyscale'],
    'balancing_method': ['undersampling', 'oversampling'],
    'until_layer': ['None', '30'],
    'batch_size': ['32', '64']
}


def plot_PR_curve(true_labels, predicted_labels, label):

    average_precision = average_precision_score(true_labels, predicted_labels)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    precision, recall, _ = precision_recall_curve(true_labels, predicted_labels)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    # plt.step(recall, precision, color='b', alpha=0.2, where='post', label=label)
    label = label + ' - AP={0:0.2f}'.format(average_precision)
    plt.step(recall, precision, where='post', label=label)
    # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.legend()


def generate_label_plot(subplot_name, dir_files):
    parts = (dir_files.split('/')[-1]).split('_')
    dict_vars = {'image_method': parts[1],
                 'balancing_method': parts[2],
                 'until_layer': parts[4],
                 'batch_size': parts[8]}

    label_list = [k for k in dict_vars.values() if k != subplot_name]
    label = '/'.join(label_list)
    return label


def main_test(list_y_axis, logger=None):

    for variable_y in list_y_axis:
        subplots_name = VARIABLES[variable_y]

        for subplot_name in subplots_name:
            regex_subplot = 'data/output/' + f'*{subplot_name}*'
            dirs_subplot = glob.glob(regex_subplot)

            plt.figure()
            # Read one output directory
            for dir_files in dirs_subplot:
                true_labels = np.genfromtxt(dir_files+'/true.csv', dtype=int, delimiter=",")
                predicted_labels = np.genfromtxt(dir_files + '/predicted.csv', dtype=int, delimiter=",")

                legend_label = generate_label_plot(subplot_name, dir_files)
                plot_PR_curve(true_labels, predicted_labels, label=legend_label)

            title = 'Precision-Recall curve: ' + variable_y + ' = ' + subplot_name
            plt.title(title)
            plt.legend()
            plt.show()


if __name__ == '__main__':

    y_axis = ['image_method', 'balancing_method', 'until_layer', 'batch_size']  # 'image_method', 'balancing_method', 'until_layer', 'batch_size'
    main_test(y_axis)

