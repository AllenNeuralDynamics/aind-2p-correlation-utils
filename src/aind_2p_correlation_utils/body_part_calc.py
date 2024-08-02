"""Module of methods for body part calculations."""

import numpy as np
import pandas as pd


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
      bodyparts_coords
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
    df.index.name = new_column_names[0]

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
      paw2_speed, paw1_speed, time (seconds)]

    """

    # Identifying body part columns
    body_parts = set([col.split("_")[0] for col in df.columns if "_" in col])

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
            # Assuming no frames are skipped, then delta_frames is just 1
            # and change of pixels per frame is just delta_pixels. Since
            # we are normalizing the velocity in the next step, we don't need
            # to convert pixels/frames to pixels/seconds here.
            df[f"{body_part}_speed"] = delta_pixels
            # Normalize by maximum speed.
            # TODO: Maybe move the normalization to a separate function to
            #  make it easier to replace if needed
            df[f"{body_part}_speed"] = (
                df[f"{body_part}_speed"] / df[f"{body_part}_speed"].max()
            )
            # Rename speed column to convey that it is now normalized.
            df.rename(
                columns={f"{body_part}_speed": f"{body_part}_norm_speed"},
                inplace=True,
            )

    # Converting the frame indices to time
    df["time (seconds)"] = df.index / frame_rate

    # Function modifies df inplace
    return None
