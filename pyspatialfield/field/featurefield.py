import numpy as np
import numpy.ma as ma
from scipy import ndimage


class FeatureField(object):
    """
    A type representing a scalar field-like, 2-D, data structure.

    """
    def __init__(self, data):
        self.data = data
        self._xmean = None
        self._ymean = None

    def __repr__(self):
        return "FeatureField ({} {})".format(self.dim[0], self.dim[1])

    def is_binary(self):
        return self.data.dtype == np.bool

    @property
    def dim(self):
        if self.data is not None:
            dim = self.data.shape
            assert len(dim) == 2
            return dim
        else:
            return 0, 0

    def spatial_moment(self, moment):
        """
        Calculates the spatial moment M_ij

        :param moment: the moment parameters i,j
        :type moment: tuple
        :return: Spatial moment
        :rtype: float
        """
        dimx, dimy = self.data.shape
        x, y = np.mgrid[:dimx, :dimy]
        xmoment, ymoment = moment
        if xmoment == 0 and ymoment >= 1:
            return np.sum((y ** ymoment) * self.data)
        elif ymoment == 0 and xmoment >= 1:
            return np.sum((x ** xmoment) * self.data)
        else:
            return np.sum((x ** xmoment * y ** ymoment) * self.data)

    def spatial_center_moment(self, moment):
        """
        Calculates the spatial center moment mu_ij

        :param moment: the moment parameters i,j
        :type moment: tuple
        :return: spatial center moment
        :rtype: float
        """
        xmean = self._xmean if self._xmean is not None else self.xmean()
        ymean = self._ymean if self._ymean is not None else self.ymean()
        dimx, dimy = self.data.shape
        x, y = np.mgrid[:dimx, :dimy]
        xmoment, ymoment = moment
        return np.sum(((x-xmean) ** xmoment * (y-ymean) ** ymoment) * self.data)

    @property
    def rad(self):
        """
        The orientation angle in radians of the feature.

        :return: angle (rad)
        :rtype: float
        """
        mcen11 = self.spatial_center_moment((1, 1))
        mcen20 = self.spatial_center_moment((2, 0))
        mcen02 = self.spatial_center_moment((0, 2))
        if mcen20 - mcen02 == 0:
            return 0.5 * np.arctan(np.inf)
        return 0.5 * np.arctan((2 * mcen11) / (mcen20 - mcen02))

    @property
    def degrees(self):
        """
        The orientation angle in degrees of the feature.

        :return: angle (degrees)
        :rtype: float
        """
        return (self.rad / np.pi) * 180

    def xmean(self):
        self._xmean = self._mean(0)
        return self._xmean

    def ymean(self):
        self._ymean = self._mean(1)
        return self._ymean

    @property
    def area(self):
        """
        Calculates the area M_00

        :return: area
        :rtype: float
        """
        return self.spatial_moment((0, 0))

    def centroid(self):
        """
        The centroid of the feature

        :return: centroid
        :rtype: tuple
        """
        if self._xmean is not None and self._ymean is not None:
            return self._xmean, self._ymean
        return self.xmean(), self.ymean()

    def _mean(self, dim):
        assert dim in (0, 1)
        m00 = self.spatial_moment((0, 0))
        m01 = self.spatial_moment((0, 1))
        m10 = self.spatial_moment((1, 0))
        mean = m10/m00 if dim == 0 else m01 / m00
        return mean

    @staticmethod
    def create_field_with_features_from_image_files(file_background, file_foreground):
        raise NotImplementedError("create_field_with_features_from_image_files() not yet implemented")

    @staticmethod
    def generate_from_single_image(decoder, threshold=50):
        """
        Generates a label map from a data source

        :param decoder: decoder of the data source
        :param threshold: cut-off value
        :return: label map
        :rtype: numpy.ndarray
        """
        array = decoder.decode()
        if len(array.shape) == 3:
            array = np.mean(array, 2)
        masked_array = np.ma.masked_where(array > threshold, array, copy=True)
        labeled_map, n = ndimage.label(masked_array.mask)
        return labeled_map

    @staticmethod
    def get_feature_from_labelmap(map, label):
        """
        Get a feature mask of a specific feature

        :param map: map with multiple features
        :type map: numpy.ndarray
        :param label: feature
        :type label: int
        :return: feature map
        :rtype: numpy.ndarray with dtype = bool
        """
        if label is not None and map.dtype == np.int32:
            features = ma.masked_where(map == label, map).mask
        elif map.dtype == np.bool:
            features = map.features
        else:
            raise TypeError("Unknown input type")
        return features

    @staticmethod
    def generate_from_image_diff(array_foreground, array_background, threshold=10):
        """
        Generate a feature map based on pixel value difference between two sources, foreground and background.

        :param array_foreground: array for foreground source
        :type array_foreground: numpy.ndarray
        :param array_background: array for background source
        :type array_background: umpy.ndarray
        :param threshold: threshold value
        :return: feature map
        :rtype: numpy.ndarray
        """
        if len(array_foreground.shape) == 3:
            array_foreground = np.mean(array_foreground, 2)
        if len(array_background.shape) == 3:
            array_background = np.mean(array_background, 2)
        diffarray = abs(array_foreground[:, :] - array_background[:, :])
        masked_array = np.ma.masked_where(diffarray < threshold, diffarray, copy=True)
        binary_array = np.invert(masked_array.mask)
        labeled_map, n = ndimage.label(binary_array)
        return labeled_map
