"""Tests bodypart calculation methods."""

import unittest
from pandas.testing import assert_frame_equal
from aind_2p_correlation_utils.bodypart_calc import adjust_coordinates, velocity

class BodyPartCalculation(unittest.TestCase):
    """Example Test Class"""

    def test_adjust_coordinates(self, csv_file, output_path):
        """Test the adjust_coordinates method."""
        csv_file = r"C:\Users\alena.lemeshova\Projects\aind-2p-correlation-utils\tests\corr_test.csv"
        output_path = r"C:\Users\alena.lemeshova\Projects\aind-2p-correlation-utils\tests"
        frame_rate = 19
        timing = 1 / frame_rate
        # Expected output
        expected_output = pd.read_csv(r"C:\Users\alena.lemeshova\Projects\aind-2p-correlation-utils\tests\adjusted_coords.csv")
        # Call the method
        output = adjust_coordinates(csv_file, output_path)
        # Compare the result with the expected output
        assert_frame_equal(output, expected_output)

    def test_velocity(self, csv_file, output_path, frame_rate, timing):
        """Test the velocity method."""
        csv_file = r"C:\Users\alena.lemeshova\Projects\aind-2p-correlation-utils\tests\corr_test.csv"
        output_path = r"C:\Users\alena.lemeshova\Projects\aind-2p-correlation-utils\tests"
        frame_rate = 19
        timing = 1 / frame_rate
        # Expected output
        expected_output = pd.read_csv(r"C:\Users\alena.lemeshova\Projects\aind-2p-correlation-utils\tests\velocity.csv")
        # Call the method
        output = velocity(csv_file, output_path, frame_rate, timing)
        # Compare the result with the expected output
        assert_frame_equal(output, expected_output)

if __name__ == "__main__":
    unittest.main()
