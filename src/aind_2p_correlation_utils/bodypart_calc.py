"""Module of methods for body part calculations."""

import numpy as np
import pandas as pd


def adjust_coordinates(trial_coords):
    """
    Method to adjust the coordinates of the body parts in a video into the
      format needed for the analysis.
    Parameters
        ----------
        csv_file : str
            The path to the csv file containing the body part coordinates.

    Returns
        -------
        df : DataFrame
            A pandas DataFrame containing the body part coordinates.
    """
    df = pd.read_csv(trial_coords, header=None)
    # Combine rows with bodypart names and their corresponding coordinates
    index0 = 0
    index1 = 1
    index2 = 2

    # Combine the values from both rows into a new row
    combined_row = df.iloc[index1] + "_" + df.iloc[index2]

    # Replace the original rows with the combined row
    df.iloc[index1] = combined_row

    # Dropping the second and the 0th rows
    df = df.drop(index2)
    df = df.drop(index0)

    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]
    df.columns = new_header

    print(df.head())

    return df


def velocity(csv_file, frame_rate, timing):
    df = pd.read_csv(csv_file)

    # Calculating average velocity in pixels/second
    for column in df[["paw1_x", "paw1_y", "paw2_x", "paw2_y"]]:
        df[f"{column}_velocity"] = df[column].diff() / timing

    # Calculating overall velocity of a body part
    df["paw1_velocity"] = np.sqrt(
        (df["paw1_x_velocity"] ** 2) + (df["paw1_y_velocity"] ** 2)
    )
    df["paw2_velocity"] = np.sqrt(
        (df["paw2_x_velocity"] ** 2) + (df["paw2_y_velocity"] ** 2)
    )

    # Calculating normalized velocity
    df["paw1_velocity"] = df["paw1_velocity"] / df["paw1_velocity"].max()
    df["paw2_velocity"] = df["paw2_velocity"] / df["paw2_velocity"].max()

    # Converting the frame indices to time for plotting purposes
    df["x_rescaled"] = df["bodyparts_coords"] / (frame_rate)

    return df
