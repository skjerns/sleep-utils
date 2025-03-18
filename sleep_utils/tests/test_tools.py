# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:57:35 2025

@author: Simon.Kern
"""

import sleep_utils
import unittest
import numpy as np
import warnings
from unittest.mock import patch
import io

from sleep_utils import hypno_summary, read_hypno


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def sanity_check_hypno(hypno):
    s = hypno_summary(hypno)
    for key, val in s.items():
        assert val>=0, f'negative value for {key}, not possible'

        # percentage cant be higher than 100%
        if key.startswith('perc_'):
            assert s[key] < 100, f'{key} has > 100%: {val}'

        # latency cant be before lights off
        if key.startswith('perc_'):
            assert s[key] < 100, f'{key} has > 100%: {val}'


class TestHypnoSummary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample hypnograms for testing
        cls.hypno1 = read_hypno('./data/test_adapt_hypno.txt')
        cls.hypno2 = read_hypno('./data/test_exe_hypno.txt')
        cls.hypno3 = read_hypno('./data/test_rest_hypno.txt')
        cls.hypno_waso = read_hypno('./data/14201_hypno.txt')
        cls.basic_hypno = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 1, 2, 3, 4, 4, 0, 0]  # Simple sleep pattern
        cls.missing_stages_hypno = [0, 0, 1, 1, 2, 2, 0, 0]  # Missing S3 and REM


    def test_missing_stages(self):
        """Test behavior when stages are missing"""
        result = hypno_summary(self.missing_stages_hypno)

        # Check if latencies for missing stages are NaN
        self.assertTrue(np.isnan(result['lat_S3']))
        self.assertTrue(np.isnan(result['lat_REM']))

    def test_sleep_cycles(self):
        """Test sleep cycle counting"""
        multi_cycle_hypno = [0, 1, 2, 3, 2, 4, 2, 3, 2, 4, 0]  # 2 NREM->REM transitions
        result = hypno_summary(multi_cycle_hypno)
        self.assertEqual(result['sleep_cycles'], 2)

    def test_print_summary(self):
        """Test print_summary parameter"""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            hypno_summary(self.basic_hypno, print_summary=True)
            output = fake_stdout.getvalue()
            self.assertTrue(len(output) > 0)

    def test_integer_division_issue(self):
        """Test the TRT calculation with non-integer minutes"""
        odd_hypno = [0] * 7   + [1, 2]# 9 epochs * 30s = 210s = 3.5 minutes
        result = hypno_summary(odd_hypno)
        # Due to integer division, TRT will be 3 instead of 3.5
        self.assertEqual(result['TRT'], 4.5)


    def test_sample_hypnos(self):
        """make some sanity checks for the hypnograms"""

        for hypno in [self.hypno1, self.hypno2, self.hypno3, self.hypno_waso,
                      self.basic_hypno, self.missing_stages_hypno]:

            summary = hypno_summary(hypno)
            sleep_utils.tools.hypno_check_summary(summary, mode='raise')

            s = AttrDict(summary)

            # Test TRT calculation (assuming epochlen=30)
            assert s.TRT == len(hypno) * 30 / 60

            # Test percentages sum to 100% (only for hypnograms with all stages)
            if 1 in hypno and 2 in hypno and 3 in hypno and 4 in hypno:
                # Due to rounding, we allow a small tolerance
                total_percentage = s.perc_S1 + s.perc_S2 + s.perc_S3 + s.perc_REM
                np.testing.assert_almost_equal(total_percentage, 100, decimal=1)

            # Test TST calculation
            expected_tst = sum(x != 0 for x in hypno) * 30 / 60
            np.testing.assert_almost_equal(s.TST, expected_tst)

            # Test sleep stage minutes
            np.testing.assert_almost_equal(s.min_S1, sum(x == 1 for x in hypno) * 30 / 60)
            np.testing.assert_almost_equal(s.min_S2, sum(x == 2 for x in hypno) * 30 / 60)
            np.testing.assert_almost_equal(s.min_S3, sum(x == 3 for x in hypno) * 30 / 60)
            np.testing.assert_almost_equal(s.min_REM, sum(x == 4 for x in hypno) * 30 / 60)

            # Test sleep efficiency calculation
            expected_se = s.TST / s.TBT
            np.testing.assert_almost_equal(s.SE, round(expected_se, 2))

            # Test latencies
            if 1 in hypno:
                first_s1 = list(hypno).index(1)
                expected_lat_s1 = (first_s1 - 0) * 30 / 60  # Assuming lights_off_epoch=0
                np.testing.assert_almost_equal(s.lat_S1, expected_lat_s1)

            # Test SQI calculation
            expected_sqi = round(s.SE * (1 - s.FI), 2)
            if expected_sqi >= 0:
                np.testing.assert_almost_equal(s.SQI, expected_sqi)
            else:
                self.assertTrue(np.isnan(s.SQI))

    def test_edge_cases(self):
        # Test empty hypnogram
        with self.assertRaises(IndexError):  # Should raise IndexError for empty array
            hypno_summary([])

        # Test all wake hypnogram
        all_wake = [0, 0, 0, 0, 0]
        with self.assertRaises(IndexError):  # No sleep onset
            hypno_summary(all_wake)

        # Test custom lights off/on
        custom_lights = [0, 0, 1, 2, 3, 4, 0, 0]
        s = AttrDict(hypno_summary(custom_lights, lights_off_epoch=1, lights_on_epoch=6))
        self.assertEqual(s.TBT, (6-1)*30/60)  # 5 epochs * 30s / 60 = 2.5 minutes

    def test_awakenings_and_fragmentation(self):
        # Test awakening count
        hypno_with_awakenings = [0, 1, 2, 0, 3, 4, 0, 2, 3, 0]
        s = AttrDict(hypno_summary(hypno_with_awakenings))
        self.assertEqual(s.awakenings, 2)

        # Test fragmentation index
        # FI = (awakenings + stage_shifts) / TST
        expected_fi = (3 + 6) / s.TST  # 3 awakenings, 6 stage shifts, divided by TST
        np.testing.assert_almost_equal(s.FI, round(expected_fi, 2))

    def test_file_waso(self):
        s = sleep_utils.hypno_summary(self.hypno_waso)

if __name__ == '__main__':
    unittest.main()
