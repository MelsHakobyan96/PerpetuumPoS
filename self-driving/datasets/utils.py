import ast
import pandas as pd
import numpy as np
import h5py


def read_csv(path):
    """
        Reading csv file into a pandas dataframe.
    """
    return pd.read_csv(path,  index_col=False)


def read_txt(path):
    """
        Reading txt file into a pandas dataframe.
    """
    f = open(path, "r")
    return f.readlines()


def list_to_numpy_array(data_list):
    """
        Stacking lists to form a numpy array.
    """
    return np.stack(data_list)


def convert_hdf5_to_numpy(data_path):
    """
        Reading hdf5 file.
    """
    hf = h5py.File(data_path, 'r')
    return hf


def str_to_list(data_list):
    """
        Converting string to its real data type.
    """
    return ast.literal_eval(data_list)


def write_in_txt(data, txt_path):
    """
        Writing data into a txt file.
    """
    with open(txt_path, 'w') as f:
        for line in data:
            f.write(line)
            f.write('\n')
