"""Tests bodypart calculation methods."""

import os
import unittest
from pathlib import Path

from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from aind_2p_correlation_utils.body_part_calc import (
    add_speed_columns,
    apply_convolution,
    apply_median_filter,
    normalize_speed,
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
        expected_index_name = "frames"
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

    def test_apply_median_filter(self) -> None:
        """Tests _apply_median_filter"""
        speed_df = self.expected_speed_df.copy(deep=True)
        apply_median_filter(speed_df)
        expected_paw1_vals = [
            11.61019506231878,
            22.10746855460816,
            11.61019506231878,
            8.72191097741752,
            8.72191097741752,
            8.150387060775728,
            8.150387060775728,
            8.150387060775728,
            10.700233740579602,
            19.55936497418573,
            19.55936497418573,
            5.789661468183895,
            5.424312090290383,
            5.789661468183895,
            5.424312090290383,
            6.803101898845965,
            26.463162275577343,
            26.463162275577343,
            19.817659913011504,
            19.817659913011504,
            19.817659913011504,
            22.41813036616135,
            22.41813036616135,
            22.41813036616135,
            15.17092512565473,
            15.17092512565473,
            19.340060895472423,
            19.340060895472423,
            15.878970476404154,
            13.637049245776923,
            13.637049245776923,
            9.229595213266528,
            9.229595213266528,
            7.498996147946154,
            5.096165473771867,
            4.518893963105363,
            4.518893963105363,
            15.29965879712601,
            15.29965879712601,
            16.651763584214756,
            16.651763584214756,
        ]

        assert_allclose(
            expected_paw1_vals,
            speed_df["paw1_speed (pixels per second)"].values,
            rtol=1e-10,
        )

    def test_apply_convolution(self) -> None:
        """Tests apply_convolution method"""
        speed_df = self.expected_speed_df.copy(deep=True)
        apply_median_filter(speed_df)
        apply_convolution(speed_df)
        expected_convolution = [
            11.61019506231878,
            33.122402878971265,
            43.03439512258013,
            49.54991028488487,
            55.73137159685746,
            61.02438225646514,
            66.04601750727619,
            70.81019090761038,
            77.87994934022144,
            93.44636881544048,
            108.21468986988278,
            108.4561279185778,
            108.31983793507891,
            108.55588498779603,
            108.41448040748628,
            109.65911552209263,
            130.4999979239147,
            150.27235744807905,
            162.38547545557176,
            173.877547604198,
            184.7804152380756,
            197.7247576322021,
            210.00543686093653,
            221.65647927986242,
            225.46296145511278,
            229.07428330992988,
            236.66958658524024,
            243.87547471219256,
            247.25082278886438,
            248.2111937838925,
            249.12232605862835,
            245.57929007535427,
            242.2179073892632,
            237.29826547603136,
            230.22802548764957,
            222.9430088692142,
            216.0314988408949,
            220.25511035961975,
            224.26217511957097,
            229.41590037276052,
            234.30539145236662,
        ]
        assert_allclose(
            expected_convolution,
            speed_df["paw1_speed (pixels per second)"].values,
            rtol=1e-10,
        )

    def test_normalize_speed(self) -> None:
        """Tests normalize_speed"""
        speed_df = self.expected_speed_df.copy(deep=True)
        apply_median_filter(speed_df)
        apply_convolution(speed_df)
        normalize_speed(speed_df)
        expected_norm_vals = [
            0.04660439409828906,
            0.13295638091937312,
            0.17274403223279328,
            0.19889791119413283,
            0.2237108671815374,
            0.2449575002848347,
            0.265114807461026,
            0.28423863901682556,
            0.3126173015978231,
            0.3751023454776544,
            0.43438374866657103,
            0.43535290326830756,
            0.43480582270088053,
            0.43575333734740634,
            0.43518572631652475,
            0.4401818064924679,
            0.5238390311641634,
            0.6032071064265594,
            0.6518302796247817,
            0.6979605174498802,
            0.7417256339947205,
            0.7936854185668998,
            0.8429811979658296,
            0.8897495571219814,
            0.905029127746874,
            0.919525306840703,
            0.9500135549053219,
            0.97893865463828,
            0.9924876132164745,
            0.9963426309911645,
            1.0,
            0.9857779266943731,
            0.9722850264823705,
            0.9525371299727897,
            0.9241565343825018,
            0.8949138055845981,
            0.8671703666979014,
            0.8841243329904723,
            0.9002090606154391,
            0.9208965892473638,
            0.9405234575291549,
        ]
        assert_allclose(
            expected_norm_vals,
            speed_df["paw1_speed (normalized)"].values,
            rtol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
