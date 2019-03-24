# Pickle Helpers
# pickle_helpers

import _pickle as pickle
import os.path


def save_object(instance, filename, folder, default_fname = "data"):

    if not filename:
        print("Please give valid string for filename.")
        print("Object not saved.")
        return

    folder_path = os.path.abspath(folder)
    if not folder_path:
        print("Please give valid path or folder name for folder.")
        print("Object not saved.")
        return

    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"

    full_path = os.path.join(folder_path, filename)

    with open(full_path, 'wb') as write:
        pickle.dump(instance, write, -1)


def load_object(filename):
    with open(filename, 'rb') as read:
        instance = pickle.load(read)

    return instance
