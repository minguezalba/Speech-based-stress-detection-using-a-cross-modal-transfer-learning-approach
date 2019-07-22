import glob


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
