import numpy as np
from numpy import linalg


class Component(object):

    def __init__(self, center, length, v):
        self.center = center
        self.length = length
        self.v = v

    def __repr__(self):
        return "center: {0}, length: {1}, direction: {2}".format(self.center, self.length, self.v)

    @staticmethod
    def calculate_eigenvector(data):
        """
        Calculate eigenvalues, eigenvector, and center vector of the principal components.
        :param data: The variables to calculate the eigenvector from .
        :return: tuple of center vector, eigenvalues and eigenvector
        """
        center_vector = np.mean(data, 0)
        cov_mat = np.cov(data.T)
        eigen_values, eigen_vector = linalg.eig(cov_mat)
        return center_vector, eigen_values, eigen_vector


class ComponentEvaluator(object):

    def __init__(self, data):
        self.component_index = 0
        self.center_vector, self.eigen_values, self.eigen_vector = Component.calculate_eigenvector(data)

    def __iter__(self):
        return self

    def __next__(self):
        index = self.component_index
        if self.component_index >= self.eigen_vector.shape[0]:
            raise StopIteration
        self.component_index += 1
        return Component(self.center_vector, np.sqrt(self.eigen_values[index]), self.eigen_vector.T[index])
