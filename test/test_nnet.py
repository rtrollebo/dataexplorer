from pathlib import Path
import os
import unittest
import numpy as np
from nnet.architecture import Architecture, Network, Layer
from nnet.operations import run


class TestNnet(unittest.TestCase):

    @staticmethod
    def create_test_data_1():
        nodes1 = [np.array([0.35, 0.9]), np.array([0.0, 0.0]), np.array([0.0])]
        weights1 = [None, np.array([[0.1, 0.4], [0.8, 0.6]]), np.array([0.3, 0.9])]
        arc1 = Architecture([Layer(2, name="Input"), Layer(2, name="Hidden"), Layer(1, name="Output")])
        return Network(arc1, nodes1, np.array([0.5]), weights1)

    @classmethod
    def setUpClass(cls):
        # Compose absolute path to make the test runnable from the unittest cli as well.
        module_path = Path(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def tearDownClass(cls):
        pass

    def test_nnet(self):
        network = TestNnet.create_test_data_1()
        error = run(network, iterations=1)
        self.assertAlmostEqual(error, np.array([-0.19028349]), delta=1E-7)

        network = TestNnet.create_test_data_1()
        error = run(network, iterations=2)
        self.assertAlmostEqual(error, np.array([-0.18198189]), delta=1E-7)

    def test_nnet_convergence(self):
        network = TestNnet.create_test_data_1()
        error = run(network, iterations=40)
        self.assertLess(abs(error), np.array([0.05]))


if __name__ == '__main__':
    unittest.main()
