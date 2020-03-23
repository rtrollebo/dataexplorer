import numpy as np
import numpy.ma as ma
import scipy
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
    def generate_from_single_image(decoder, threshold=50, preprocess=None, decoder_callback=None):
        """
        Generates a label map from a data source

        :param decoder: Decoder containing the image data.
        :param threshold: Pixel intensity threshold
        :param preprocess: Preprocessor function
        :param decoder_callback: Decoder callback
        :return: Tuple of decoded image and map with labelled features.
        """
        img_array = decoder.decode(callback=decoder_callback)
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, 2)
        if preprocess is not None:
            array = preprocess(img_array)
        else:
            array = img_array
        masked_array = np.ma.masked_where(array > threshold, array, copy=True)
        labeled_map, n = ndimage.label(masked_array.mask)
        if preprocess is not None:
            scipy.misc.imsave("testimg_masked.png", masked_array.mask.astype(int))
        return (img_array, labeled_map)

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

    @staticmethod
    def get_features_by_size(map):
        unique = np.unique(map, return_counts=True)
        return iter(sorted(list(zip(unique[0], unique[1])), key=lambda x: x[1], reverse=True))


class EllipseFeature(object):

    def __init__(self, semi_major_axis=None, semi_minor_axis=None):
        self.semi_major_axis = semi_major_axis
        self.semi_minor_axis = semi_minor_axis

    def __repr__(self):
        return "Semi-major axis: {0}, \nSemi-minor axis: {1}".format(self.semi_major_axis, self.semi_minor_axis)

    def from_feature_field(self, field: FeatureField):
        """
        Calculates the ellipse parameters based on a FeatureField
        :param field: The feature field from which to calculate the ellipsis.
        :param type: FeatureField
        :return:
        """
        self.from_moments(
            field.spatial_center_moment((2, 0)),
            field.spatial_center_moment((0, 2)),
            field.spatial_center_moment((1, 1)))

    def from_moments(self, mu20, mu02, mu11):
        """
        Calculates the ellipse parameters based on the mu_20, mu_02 and mu_11 spatial central moments.
        :param mu20:
        :param mu02:
        :param mu11:
        :return:
        """
        emax = self._eigv_max(mu20, mu02, mu11)
        emin = self._eigv_min(mu20, mu02, mu11)
        self.semi_major_axis = self._semiaxis_maj(emax, emin)
        self.semi_minor_axis = self._semiaxis_min(emax, emin)

    def _eigv_max(self, mu20, mu02, mu11):
        return 0.5 * (mu20 + mu02 + np.sqrt((4 * (mu11 ** 2.)) + ((mu20 - mu02) ** 2.)))

    def _eigv_min(self, mu20, mu02, mu11):
        return 0.5 * (mu20 + mu02 - np.sqrt((4 * (mu11 ** 2.)) + ((mu20 - mu02) ** 2.)))

    def _semiaxis_maj(self, emax, emin):
        return ((4. / np.pi) ** (1. / 4.)) * (((emax ** 3) / emin) ** (1. / 8.))

    def _semiaxis_min(self, emax, emin):
        return ((4. / np.pi) ** (1. / 4.)) * (((emin ** 3) / emax) ** (1. / 8.))
