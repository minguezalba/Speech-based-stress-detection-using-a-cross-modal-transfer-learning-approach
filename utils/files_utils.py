import glob
import os

from datetime import datetime


def create_dir_logs(image_method, until_layer, n_epochs, batch_size):
    dir_experiment = '_'.join((
        str(int(datetime.timestamp(datetime.now()))), image_method.split('/')[0], image_method.split('/')[1], 'until',
        str(until_layer), 'epochs', str(n_epochs), 'batch', str(batch_size)))

    root_output = 'logs/' + dir_experiment + '/'
    os.mkdir(root_output)

    return root_output


def read_directory(folder_name, extension):
    """
    This method reads all the files from a specific directory with a specific extension and returns a list of tuples,
    each one corresponding to a file with the format (path, filename, extension)
    :return: List of tuples/files with format (path, filename, extension)
    """
    files = []
    for f in glob.glob(f"{folder_name}*.{extension}"):
        parts = f.split('/')
        path = f
        filename = parts[-1].split('.')[0]

        files.append((path, filename))

    print(f'Number of files in {folder_name}: {len(files)}')
    return sorted(files)
