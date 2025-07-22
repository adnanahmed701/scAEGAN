import copy

import pandas as pd
import numpy as np
from keras.utils import to_categorical

def load_data(path, input_shape, label_col=-1, num_classes=10):
    df = pd.read_csv(path, index_col=0)

    # Drop rows with NaN in the label column
    df = df.dropna(subset=[df.columns[label_col]])

    features = df.drop(df.columns[label_col], axis=1)
    data = features.values.reshape((-1,) + input_shape)

    labels = df.iloc[:, label_col].astype(int).values

    labels = to_categorical(labels, num_classes=num_classes)

    return data, labels





def minibatch(data, labels, batch_size):
    data = copy.copy(data)
    labels = copy.copy(labels)
    length = data.shape[0]
    indices = np.arange(length)
    np.random.shuffle(indices)
    epoch = 0
    i = 0

    while True:
        if i + batch_size > length:
            np.random.shuffle(indices)
            i = 0
            epoch += 1
        batch_indices = indices[i:i + batch_size]
        data_batch = data[batch_indices]
        label_batch = labels[batch_indices]
        i += batch_size
        yield epoch, data_batch, label_batch



def minibatchAB(dataA, labelA, dataB, labelB, batch_size):
    batchA = minibatch(dataA, labelA, batch_size)
    batchB = minibatch(dataB, labelB, batch_size)

    while True:
        ep1, A, A_labels = next(batchA)
        ep2, B, B_labels = next(batchB)
        yield max(ep1, ep2), A, B, A_labels, B_labels



