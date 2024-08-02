"""Module for methods in reading and writing data"""

from pathlib import Path

from pandas import DataFrame, read_csv


def read_trial_coordinates(trial_coords: Path) -> DataFrame:
    """
    Read a file containing trial coordinates and returns a pandas DataFrame.
    The csv file is expected to look like:
    scorer,heatmap_tracker,...,heatmap_tracker
    bodyparts,paw1,paw1,paw1,paw2,paw2,paw2
    coords,x,y,likelihood,x,y,likelihood
    0,633.45,326.82,0.99,417.58,328.60,0.99
    1,633.34,327.42,0.99,418.41,328.23,0.99

    Parameters
    ----------
    trial_coords : Path
      Location of the file containing the trial coordinates

    Returns
    -------
    DataFrame

      The output df might look like:
      scorer    heatmap_tracker
      bodyparts            paw1                       paw2
      coords                  x       y likelihood       x       y likelihood
      0                  633.45  326.82       0.99  417.58  328.60       0.99
      1                  633.34  327.42       0.99  418.41  328.23       0.99
    """

    df = read_csv(trial_coords, header=[0, 1, 2], index_col=0)

    return df


def read_speed_coordinates(speed_coords: Path) -> DataFrame:
    """
    Read a file containing speed coordinates and returns a pandas DataFrame.
    Parameters
    ----------
    speed_coords : Path
      Location of the file containing the speed coordinates

    Returns
    -------
    DataFrame

    """

    df = read_csv(speed_coords, index_col=0)

    return df
