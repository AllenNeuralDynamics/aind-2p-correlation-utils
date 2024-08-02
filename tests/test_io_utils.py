"""Tests methods in io_utils module."""

import os
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_equal
from pandas import MultiIndex
from pandas.testing import assert_index_equal

from aind_2p_correlation_utils.io_utils import (
    read_speed_coordinates,
    read_trial_coordinates,
)

TEST_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
TRIAL_COORD_FILE = TEST_DIR / "resources" / "corr_test.csv"
EXPECTED_SPEED_DF = TEST_DIR / "resources" / "expected_speed_df.csv"


class TestIoUtils(unittest.TestCase):
    """Tests methods in io_utils module"""

    def test_read_trial_coordinates(self):
        """Tests read_trial_coordinates method."""
        df = read_trial_coordinates(TRIAL_COORD_FILE)
        expected_columns = MultiIndex(
            levels=[
                ["heatmap_tracker"],
                ["paw1", "paw2"],
                ["likelihood", "x", "y"],
            ],
            codes=[[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 2, 0, 1, 2, 0]],
            names=["scorer", "bodyparts", "coords"],
        )
        expected_first_row = [
            [
                633.4585571289062,
                326.8269348144531,
                0.9944937229156494,
                417.5814514160156,
                328.6043701171875,
                0.9950076937675476,
            ]
        ]

        assert_index_equal(expected_columns, df.columns)
        assert_equal(expected_first_row, df[0:1].values.tolist())

    def test_read_speed_coordinates(self):
        """Tests read_speed_coordinates method"""
        df = read_speed_coordinates(EXPECTED_SPEED_DF)
        expected_columns = [
            "paw1_x",
            "paw1_y",
            "paw1_likelihood",
            "paw2_x",
            "paw2_y",
            "paw2_likelihood",
            "paw2_norm_speed",
            "paw1_norm_speed",
            "time (seconds)",
        ]
        expected_first_row = [
            [
                633.4585571289062,
                326.8269348144531,
                0.9944937229156494,
                417.5814514160156,
                328.6043701171875,
                0.9950076937675476,
                np.nan,
                np.nan,
                0.0,
            ]
        ]
        self.assertEqual(expected_columns, df.columns.values.tolist())
        self.assertEqual("bodyparts_coords", df.index.name)
        assert_equal(expected_first_row, df[0:1].values.tolist())


if __name__ == "__main__":
    unittest.main()
