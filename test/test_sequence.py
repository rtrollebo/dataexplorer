import datetime
from pathlib import Path
import os
import unittest
import numpy as np
from sequence import discrete
from explorer_util.datasource import DataSource, load_excel_doc_from_buf, get_columns_from_xlsx_workbook


class TestSequence(unittest.TestCase):

    dataset_2, dataset_2_xlsx, dataset_3 = None, None, None

    @classmethod
    def setUpClass(cls):
        # Compose absolute path to make the test runnable from the unittest cli as well.
        module_path = Path(os.path.dirname(os.path.abspath(__file__)))
        converter_datetime = lambda x: datetime.datetime.strptime(x.decode("utf-8"), '%Y-%m-%d %H:%M:%S')
        dataset_2 = np.genfromtxt(module_path / "dataset2.csv", delimiter=',', converters={0: converter_datetime})
        wb = load_excel_doc_from_buf(DataSource.read_from_file_source(module_path / "dataset_2.xlsx"))
        cls.dataset_2 = np.array([[i[0] for i in dataset_2], [i[1] for i in dataset_2]]).T
        cls.dataset_2_xlsx = get_columns_from_xlsx_workbook(wb)
        cls.dataset_3 = np.array([np.arange(0, 1000), (np.random.normal(0, 5, 1000) + np.arange(0, 1000))]).T
        a = 0

    @classmethod
    def tearDownClass(cls):
        pass

    def test_filter_by_segment(self):
        s1 = discrete.Series(TestSequence.dataset_3)
        s1.filter_by_segment(100, 200)
        coeff = s1.linear_coefficients()
        self.assertEqual(s1.data.shape[0], 99)
        self.assertAlmostEqual(coeff[0], 1, delta=0.1)

    def test_filter_by_datetime(self):
        ts1 = discrete.Series(TestSequence.dataset_2)
        ts1.filter_by_segment(lower=datetime.datetime(2023, 1, 1, 0, 0, 0),
                              upper=datetime.datetime(2023, 2, 28, 0, 0, 0))
        ts2 = discrete.Series(TestSequence.dataset_2)
        ts2.filter_by_segment(lower=datetime.datetime(2023, 3, 1, 0, 0, 0),
                              upper=datetime.datetime(2023, 5, 1, 0, 0, 0))
        self.assertEqual(len(ts1.data), 0)
        self.assertEqual(len(ts2.data), 2)

    def test_filter_by_datetime_xlsx(self):
        ts1 = discrete.Series(TestSequence.dataset_2_xlsx)
        ts1.filter_by_segment(lower=datetime.datetime(2023, 1, 1, 0, 0, 0),
                              upper=datetime.datetime(2023, 2, 28, 0, 0, 0))
        ts2 = discrete.Series(TestSequence.dataset_2_xlsx)
        ts2.filter_by_segment(lower=datetime.datetime(2023, 3, 1, 0, 0, 0),
                              upper=datetime.datetime(2023, 5, 1, 0, 0, 0))
        self.assertEqual(len(ts1.data), 0)
        self.assertEqual(len(ts2.data), 2)

    def test_datetime_sequence_rate(self):
        ts = discrete.Series(TestSequence.dataset_2)
        ts.to_epoch_seconds()
        coeff = ts.linear_coefficients()
        rate_per_second = 10. * (1. / (30.5 * 24 * 60 * 60))
        self.assertAlmostEqual(coeff[0], rate_per_second)


if __name__ == '__main__':
    unittest.main()