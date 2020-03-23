import unittest
import numpy as np
from pyspatialfield.field.featurefield import FeatureField

class TestSpatialMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # test data
        cls.dataset_1 = np.full((10, 10), False, dtype=bool)
        cls.dataset_1[3:6, 3:8] = True
        cls.dataset_2 = np.full((10, 10), False, dtype=bool)
        cls.dataset_2[3:8, 3:6] = True

        # Rectangle feature tilted 45 degrees
        cls.dataset_3 = np.full((10, 10), False, dtype=bool)
        cls.dataset_3[0, 2] = True
        cls.dataset_3[1, 1:4] = True
        cls.dataset_3[2, 0:5] = True
        cls.dataset_3[3, 1:6] = True
        cls.dataset_3[4, 2:7] = True
        cls.dataset_3[5, 3:8] = True
        cls.dataset_3[6, 4:7] = True
        cls.dataset_3[7, 5:6] = True

        cls.dataset_4 = np.full((10, 10), False, dtype=bool)
        cls.dataset_4[0, 6:8] = True
        cls.dataset_4[1, 6:10] = True
        cls.dataset_4[2, 5:10] = True
        cls.dataset_4[3, 5:9] = True
        cls.dataset_4[4, 4:9] = True
        cls.dataset_4[5, 4:8] = True
        cls.dataset_4[6, 3:8] = True
        cls.dataset_4[7, 3:7] = True
        cls.dataset_4[8, 5:7] = True


    @classmethod
    def tearDownClass(cls):
        pass

    def test_spatial_moment_basic(self):
        field_1 = FeatureField(TestSpatialMoments.dataset_1)
        field_2 = FeatureField(TestSpatialMoments.dataset_2)
        xmean_1 = field_1.xmean()
        ymean_1 = field_1.ymean()
        xmean_2 = field_2.xmean()
        ymean_2 = field_2.ymean()
        self.assertEqual(xmean_1, 4)
        self.assertEqual(ymean_1, 5)
        self.assertEqual(xmean_2, 5)
        self.assertEqual(ymean_2, 4)

    def test_spatial_radians_expect45degree(self):
        field = FeatureField(TestSpatialMoments.dataset_3)
        degr = field.degrees
        self.assertAlmostEqual(degr, 45.0, delta=0.01)

    def test_spatial_centroid(self):
        field = FeatureField(TestSpatialMoments.dataset_4)
        centroid = field.centroid()
        self.assertEqual(centroid, (4, 6))

    def test_spatial_area(self):
        field = FeatureField(TestSpatialMoments.dataset_4)
        area = field.area
        self.assertEqual(area, np.sum(field.data))

    def test_get_feature_by_size(self):

        pass


if __name__ == '__main__':
    unittest.main()


