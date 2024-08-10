"""Module of methods for body part calculations."""

import re

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

BODY_PART_COLUMN_PATTERN = re.compile(r"(.*)_.*$")


def rename_columns(df: pd.DataFrame) -> None:
    """
    Renames the columns of a pandas DataFrame to make it usable in a method
    downstream.
    Parameters
    ----------
    df : pd.DataFrame
      A pandas DataFrame with a MultiIndex. Might look like:
      scorer    heatmap_tracker
      bodyparts            paw1                       paw2
      coords                  x       y likelihood       x       y likelihood
      0                  633.45  326.82       0.99  417.58  328.60       0.99
      1                  633.34  327.42       0.99  418.41  328.23       0.99

    Returns
    -------
    None
      Renames the DataFrame columns inplace to conserve memory. Output should
      look like:
                        paw1_x  paw1_y  paw1_l...d  paw2_x  paw2_y  paw2_l...d
      frames
      0                 633.45  326.82        0.99  417.58  328.60        0.99
      1                 633.34  327.42        0.99  418.41  328.23        0.99

    """

    new_column_names = list()
    # First column name
    new_column_names.append("_".join(df.columns.names[1:3]))

    # Remainder of the column names
    for old_column_name in df.columns.values:
        new_name = "_".join(old_column_name[1:3])
        new_column_names.append(new_name)

    df.columns = new_column_names[1:]
    df.index.name = "frames"

    # Function modifies df inplace
    return None


def add_speed_columns(df: pd.DataFrame, frame_rate: float) -> None:
    """
    Takes a DataFrame with columns of body part pixel positions indexed by
    frame, and then adds a column with a normalized speed for each body
    part and a column that converts frames to seconds.

    Parameters
    ----------
    df : pd.DataFrame
      A DataFrame with columns that might look like:
      [paw1_x, paw1_y, paw1_likelihood, paw2_x, paw2_y, paw2_likelihood]
    frame_rate : float
      Frames per second of the camera. Will be used to convert frames into
      seconds to help align with videos taken with cameras using a different
      frame rate

    Returns
    -------
    None
      Appends the new columns to the df inplace. The new columns might look
      like:
      [paw1_x, paw1_y, paw1_likelihood, paw2_x, paw2_y, paw2_likelihood,
      paw2_speed (pixels per second), paw1_speed (pixels per second),
      time (seconds)]

    """

    # Identifying body part columns
    body_parts = set(
        [
            re.match(BODY_PART_COLUMN_PATTERN, c).group(1)
            for c in df.columns
            if re.match(BODY_PART_COLUMN_PATTERN, c)
        ]
    )
    time_series = df.index / frame_rate
    df["time (seconds)"] = time_series
    delta_t = df["time (seconds)"].diff()

    for body_part in body_parts:
        x_col = f"{body_part}_x"
        y_col = f"{body_part}_y"

        # Calculate velocities if both x and y columns are present
        if x_col in df.columns and y_col in df.columns:
            delta_x = df[x_col].diff()
            delta_y = df[y_col].diff()
            # TODO: if we want to compute a velocity (speed and direction),
            #  we'll need to use a different calculation. This computes speed.
            delta_pixels = np.sqrt((delta_x**2) + (delta_y**2))
            df[f"{body_part}_speed (pixels per second)"] = (
                delta_pixels / delta_t
            )

    # Function modifies df inplace
    return None


def apply_median_filter(df: pd.DataFrame, window_size: int = 3) -> None:
    """
    For each column in the dataframe with _speed in the name, it will apply a
    median filter and modify the column in place.
    Parameters
    ----------
    df : pd.DataFrame
    window_size : int
      Default is 3.

    Returns
    -------
    None

    """
    speed_columns = [c for c in df.columns if "_speed" in c]
    for speed_column in speed_columns:
        filtered = median_filter(df[speed_column], size=window_size)
        df[speed_column] = filtered

    # Function modifies df inplace
    return None


def apply_convolution(df: pd.DataFrame, tau: float = 1.0) -> None:
    """
    For each column in the dataframe with _speed in the name, it will convolve
    it with an exponential decay kernel. The dataframe should have a time
    series column named 'time (seconds)'.
    Parameters
    ----------
    df : pd.DataFrame
    tau : float
      Decay factor. Default is 1.0

    Returns
    -------
    None
      Modifies the columns in place.

    """
    speed_columns = [c for c in df.columns if "_speed" in c]
    time_axis = df["time (seconds)"]
    kernel = np.exp(-time_axis / tau)
    for speed_column in speed_columns:
        speed = df[speed_column]
        convolved = np.convolve(speed, kernel, mode="full")
        convolved = convolved[: len(speed)]
        df[speed_column] = convolved

    # Function modifies df inplace
    return None


def normalize_speed(df: pd.DataFrame) -> None:
    """
    For each column in the dataframe with _speed in the name, it will add a
    new column that is the column divided by its max value.
    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    None
      Appends new columns to df in place.

    """
    speed_columns = [c for c in df.columns if "_speed" in c]
    for speed_column in speed_columns:
        max_speed = df[speed_column].max()
        col_name = speed_column.replace("(pixels per second)", "(normalized)")
        df[col_name] = df[speed_column] / max_speed

    # Function modifies df inplace
    return None
