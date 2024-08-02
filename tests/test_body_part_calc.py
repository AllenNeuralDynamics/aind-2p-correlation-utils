"""Tests bodypart calculation methods."""

import os
import unittest
from pathlib import Path

from pandas.testing import assert_frame_equal

from aind_2p_correlation_utils.body_part_calc import (
    add_speed_columns,
    rename_columns,
)
from aind_2p_correlation_utils.io_utils import (
    read_speed_coordinates,
    read_trial_coordinates,
)

TEST_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
TRIAL_COORD_FILE = TEST_DIR / "resources" / "corr_test.csv"
EXPECTED_SPEED_DF = TEST_DIR / "resources" / "expected_speed_df.csv"


class TestBodyPartCalculation(unittest.TestCase):
    """Tests methods in bodypart_calc module."""

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up class. Reads csv files and sets frame rate."""
        cls.expected_speed_df = read_speed_coordinates(EXPECTED_SPEED_DF)
        cls.trial_coord_df = read_trial_coordinates(TRIAL_COORD_FILE)
        cls.frame_rate = 19

    def test_rename_columns(self):
        """Test the rename columns method."""

        # Because the method modifies the df inplace, we copy it
        trial_coord_df = self.trial_coord_df.copy(deep=True)
        rename_columns(trial_coord_df)
        expected_column_names = [
            "paw1_x",
            "paw1_y",
            "paw1_likelihood",
            "paw2_x",
            "paw2_y",
            "paw2_likelihood",
        ]
        expected_index_name = "bodyparts_coords"
        self.assertEqual(
            expected_column_names, trial_coord_df.columns.values.tolist()
        )
        self.assertEqual(expected_index_name, trial_coord_df.index.name)

    def test_add_speed_columns(self):
        """Tests add_speed_columns method."""
        # Because the method modifies the df inplace, we copy it
        trial_coord_df = self.trial_coord_df.copy(deep=True)
        expected_speed_df = self.expected_speed_df.copy(deep=True)
        rename_columns(trial_coord_df)
        add_speed_columns(trial_coord_df, frame_rate=self.frame_rate)
        # We don't care much about the order of the column names so mark
        # check_like to True
        assert_frame_equal(expected_speed_df, trial_coord_df, check_like=True)


if __name__ == "__main__":
    unittest.main()
