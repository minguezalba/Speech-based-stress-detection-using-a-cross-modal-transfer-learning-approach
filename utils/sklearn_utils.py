import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels
from utils.pytorch_utils import LABELS


def save_metrics(filepath, cm, cm_nor, f_score, precision, recall, accuracy, time_train, time_test):

    parts = filepath.split('_')
    ts = str(datetime.fromtimestamp(int(parts[0].split('/')[-1])))
    image_method = parts[1]
    balance_method = parts[2]
    until_layer = int(parts[4])
    n_epochs = int(parts[6])
    batch_size = int(parts[8].split('.')[0])

    workbook_name = 'logs/results.xlsx'
    book = load_workbook(workbook_name)
    sheet = book.active

    row = [ts, image_method, balance_method, until_layer, n_epochs, batch_size,
           cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1],
           cm_nor[0, 0], cm_nor[0, 1], cm_nor[1, 0], cm_nor[1, 1],
           f_score, precision, recall, accuracy, time_train, time_test]

    sheet.append(row)

    book.save(workbook_name)


def get_performance(total_true_labels, total_est_labels, filepath, logger, time_train='', time_test=''):

    logger.info("================================================")
    logger.info("Performance metrics:")
    logger.info("================================================")
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    cm = plot_confusion_matrix(total_true_labels, total_est_labels, classes=np.array(LABELS), logger=logger,
                               title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    cm_nor = plot_confusion_matrix(total_true_labels, total_est_labels, classes=np.array(LABELS), normalize=True,
                                   logger=logger, title='Normalized confusion matrix')

    f_score = f1_score(total_true_labels, total_est_labels)
    precision = precision_score(total_true_labels, total_est_labels)
    recall = recall_score(total_true_labels, total_est_labels)

    logger.info(f'Precision: {precision:.2f}')
    logger.info(f'Recall: {recall:.2f}')
    logger.info(f'F-Score: {f_score:.2f}')

    accuracy = (cm[0, 0] + cm[1, 1]) / len(total_true_labels)
    logger.info(f'Accuracy: {precision:.2f}')

    save_metrics(filepath, cm, cm_nor, f_score, precision, recall, accuracy, time_train, time_test)


def plot_confusion_matrix(y_true, y_pred, classes, logger,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function logger_.infos and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix")
    else:
        logger.info('Confusion matrix, without normalization')

    logger.info(cm)

    # fig, ax = plt.subplots()
    # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # # We want to show all ticks...
    # ax.set(xticks=np.arange(cm.shape[1]),
    #        yticks=np.arange(cm.shape[0]),
    #        # ... and label them with the respective list entries
    #        xticklabels=classes, yticklabels=classes,
    #        title=title,
    #        ylabel='True label',
    #        xlabel='Predicted label')
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, format(cm[i, j], fmt),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    return cm
