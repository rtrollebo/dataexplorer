from pathlib import Path
import os
import unittest
import numpy as np
from numpy import testing
from pycomponents import component
from explorer_util import datasource


class TestPrincipalComponents(unittest.TestCase): 

    dataset_1 = None

    @classmethod
    def setUpClass(cls):
        # Compose absolute path to make the test runnable from the unittest cli as well.
        module_path = Path(os.path.dirname(os.path.abspath(__file__)))
        cls.dataset_1 = datasource.DataSource().load(module_path / "dataset1.csv")

    @classmethod
    def tearDownClass(cls):
        pass

    def test_calculate_principal_components(self):
        """
        Test principal component analysis on the example given in
        Rencher, A. C. (2002), Methods of Multivariate analysis, p.384
        """
        component_evaluator = component.ComponentEvaluator(TestPrincipalComponents.dataset_1)
        components = list(component_evaluator)
        testing.assert_allclose(components[0].v, np.array([0.825, 0.565]), 0.001)
        testing.assert_allclose(components[1].v, np.array([-0.565, 0.825]), 0.001)


if __name__ == '__main__':
    unittest.main()
