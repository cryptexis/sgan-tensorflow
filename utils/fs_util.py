import os


def make_dir(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def file_exists(file_path):

    if os.path.exists(file_path):
        return True
    else:
        return False