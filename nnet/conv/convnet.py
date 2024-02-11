"""
nnet architecture tools
"""
import doctest
import numpy as np


class ConvNet:

    def __init__(self, input_size, filter_size, padding, stride) -> None:
        """

        :param input_size: Input volume size
        :param filter_size: Filter size
        :param padding: Padding
        :param stride: Stride
        """
        self.input_size = input_size
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride


def im2col(x: np.ndarray, architecture: ConvNet):
    """

    :param x:
    :param architecture:
    :return:

    >>> a1 = ConvNet(7, 3, 0, 1)
    >>> x1 = np.arange(0, a1.input_size*a1.input_size*3).reshape(a1.input_size, a1.input_size, 3)
    >>> im2col(x1, a1).shape[0] # expects 3*3*3
    27
    >>> a2 = ConvNet(227, 11, 0, 4)
    >>> x2 = np.arange(0, a2.input_size*a2.input_size*3).reshape(a2.input_size, a2.input_size, 3)
    >>> im2col(x2, a2).shape[0] # expects 11*11*3
    363
    """
    return x[0:architecture.filter_size, 0:architecture.filter_size, 0:3].flatten()


def output_spatial_size(architecture: ConvNet):
    """

    :param architecture: nnet architecture
    :return:

    >>> output_spatial_size(ConvNet(7, 3, 0, 1))
    5
    >>> output_spatial_size(ConvNet(7, 3, 0, 2))
    3

    """
    output_dim = ((architecture.input_size-architecture.filter_size+2*architecture.padding)/architecture.stride)+1
    if output_dim.as_integer_ratio()[0] != output_dim:
        raise ValueError("Invalid hyperparameters: (input_size={0}, filter_size={1}, padding={2}, stride={3})"
                         .format(architecture.input_size, architecture.filter_size, architecture.padding,
                                 architecture.stride))
    return int(output_dim)


if __name__ == "__main__":
    doctest.testmod()
