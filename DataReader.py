import os
import pickle
import numpy as np
import tarfile

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
        
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    with tarfile.open(data_dir) as tar:
        tar.extractall(path=os.path.dirname(data_dir))
    data_dir = os.path.join(os.path.dirname(data_dir), 'cifar-10-batches-py')

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Load training batches
    for i in range(1, 6):
        with open(os.path.join(data_dir, 'data_batch_{}'.format(i)), 'rb') as file:##
            batch = pickle.load(file, encoding='latin1')
            x_train.append(batch['data'])
            y_train.append(batch['labels'])

    # Load test batch
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        x_test = batch['data']
        y_test = batch['labels']

    # Convert to numpy arrays and reshape
    x_train = np.concatenate(x_train).reshape(-1, 3072).astype(np.float32)
    y_train = np.concatenate(y_train).astype(np.int32)
    x_test = np.array(x_test).reshape(-1, 3072).astype(np.float32)
    y_test = np.array(y_test).astype(np.int32)

    return x_train, y_train, x_test, y_test
    ### YOUR CODE HERE


def train_valid_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid