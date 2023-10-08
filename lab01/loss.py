import numpy as np


def cross_entropy_loss(AL, Y):
    """
    Implement the cross-entropy loss function.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    loss -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    loss = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    loss = np.squeeze(loss)
    assert (loss.shape == ())

    return loss
