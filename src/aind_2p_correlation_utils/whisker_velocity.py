import pandas as pd
import numpy as np


def velocity(csv_file, frame_rate, timing):
    df = pd.read_csv(csv_file)

    # Identifying body part columns
    body_parts = set(
        [col.rsplit("_", 1)[0] for col in df.columns[7:] if "_" in col]
    )
    print(body_parts)

    for body_part in body_parts:
        x_col = f"{body_part}_x"
        y_col = f"{body_part}_y"

        # Calculate velocities if both x and y columns are present
        if x_col in df.columns and y_col in df.columns:
            df[f"{x_col}_velocity"] = df[x_col].diff() / timing
            df[f"{y_col}_velocity"] = df[y_col].diff() / timing

            df[f"{body_part}_velocity"] = np.sqrt(
                (df[f"{x_col}_velocity"] ** 2) + (df[f"{y_col}_velocity"] ** 2)
            )
            df[f"{body_part}_velocity"] = (
                df[f"{body_part}_velocity"] / df[f"{body_part}_velocity"].max()
            )

    # Converting the frame indices to time for plotting purposes
    df["x_rescaled"] = df.index / frame_rate

    # Saving the velocity calculations into a csv file for further analysis
    # df.to_csv(f'{output_path}/velocity.csv', index = False)

    return df
