import numpy as np
import scipy


def coocc_matrix_Label(one_hot_label):
    """
    Create Labels co-occurrences matrix

    Parameters :
        one_hot_label : DataFrame
            One hot encoding DataFrame with only labels

    Returns :
        coocc : DataFrame
            The co-occurrence matrix
    """

    coocc = scipy.linalg.blas.dgemm(
        alpha=1.0, a=one_hot_label.T, b=one_hot_label.T, trans_b=True
    )
    np.fill_diagonal(coocc, 0)  # replace the diagonal by 0

    return coocc
