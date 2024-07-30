"""Module of methods for body part calculations."""
import numpy as np
import pandas as pd

def adjust_coordinates(trial_coords):
    """
    Method to adjust the coordinates of the body parts in a video into the format needed for the analysis.
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
    # Combine the rows that contain the names for the bodyparts and their corresponding coordinates
    index0 = 0
    index1 = 1
    index2 = 2

    # Combine the values from both rows into a new row
    combined_row = df.iloc[index1] + '_' + df.iloc[index2]

    # Replace the original rows with the combined row
    df.iloc[index1] = combined_row

    # Dropping the second and the 0th rows
    df = df.drop(index2)
    df = df.drop(index0)

    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] 
    df.columns = new_header

    print(df.head())

    return df


def velocity(csv_file, frame_rate, timing):
    df = pd.read_csv(csv_file)
   
    # Identifying body part columns
    body_parts = set([col.split('_')[0] for col in df.columns if '_' in col])
   
    for body_part in body_parts:
        x_col = f'{body_part}_x'
        y_col = f'{body_part}_y'
       
        # Calculate velocities if both x and y columns are present
        if x_col in df.columns and y_col in df.columns:
            df[f'{x_col}_velocity'] = df[x_col].diff() / timing
            df[f'{y_col}_velocity'] = df[y_col].diff() / timing
           
            df[f'{body_part}_velocity'] = np.sqrt((df[f'{x_col}_velocity'] ** 2) + (df[f'{y_col}_velocity'] ** 2))
            df[f'{body_part}_velocity'] = df[f'{body_part}_velocity'] / df[f'{body_part}_velocity'].max()
            
    # Converting the frame indices to time for plotting purposes
    df['x_rescaled'] = df.index / frame_rate
   
    return df


# def velocity_graph(csv_file, output_path):
#     df = pd.read_csv(csv_file)
    
#     body_parts = set([col.split('_')[0] for col in df.columns if '_' in col])
    
#     for body_part in body_parts:
#         x_col = f'{body_part}_x'
#         y_col = f'{body_part}_y'
    
#     for column in df[f'{body_part}_velocity']:
#         plt.figure()
#         plt.plot(df['x_rescaled'], df[column])
#         plt.ylabel('Normalized velocity')
#         plt.xlabel('Time (s)')
#         plt.grid(True)
#         plt.savefig(f'/{output_path}/{bodypart}.png')
        
#     return df