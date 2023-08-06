import time
import numpy as np
from numpy import linalg


class Series(object):

    def __init__(self, data: np.ndarray):
        self.data = data

    def filter_by_segment(self, lower,
                          upper):

        """
        Filter by lower and upper bound of indices
        """
        shp = self.data.shape
        data = self.data
        if shp[1] != 2:
            data = self.data.reshape(int(shp[0] / 2), 2)
        mask = (data[:, 0] > lower) \
               & (data[:, 0] < upper)
        self.data = data[mask]

    def to_epoch_seconds(self):
        f = np.vectorize(lambda x: time.mktime(x.timetuple()))
        # self.data = np.array([f(self.data.T[0]), self.data.T[1]]).T
        self.data = np.array([f(self.data.T[0]), self.data.T[1]]).T

    def linear_coefficients(self, method='lstsq'):
        if method == 'lstsq':
            x = self.data.T[0]
            y = self.data.T[1]
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A.astype(float), y.astype(float), rcond=None)[0]
            return m, c
        else:
            raise ValueError("Unsupported method " + method)
