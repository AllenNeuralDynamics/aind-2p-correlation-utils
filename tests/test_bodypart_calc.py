"""Tests bodypart calculation methods."""

import os
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from aind_2p_correlation_utils.bodypart_calc import adjust_coordinates, velocity
from pathlib import Path
from aind_2p_correlation_utils.io_utils import trial_coordinates, velocity_coordinates

TEST_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
RESOURCES_DIR = TEST_DIR / "resources"

trial_coords = RESOURCES_DIR / "corr_test.csv"
adj_coords = RESOURCES_DIR / "adjusted_coords.csv"

frame_rate = 19
timing = 1 / frame_rate

class TestBodyPartCalculation(unittest.TestCase):

    def test_adjust_coordinates(self):
        """Test the adjust_coordinates method."""
        
        # Expected output
        expected_output = pd.read_csv(trial_coords, header=None) 
        # Call the method
        output = adjust_coordinates(trial_coordinates(trial_coords))
        # Compare the result with the expected output
        assert_frame_equal(output, expected_output)

    def test_velocity(self):
        """Test the velocity method."""
        
        # Expected output
        expected_output = pd.read_csv(adj_coords)
        # Call the method
        output = velocity(df=velocity_coordinates(adj_coords), frame_rate=frame_rate, timing=timing)  
        # Compare the result with the expected output
        assert_frame_equal(output, expected_output)

if __name__ == "__main__":
    unittest.main()
