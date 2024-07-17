"""Module of methods for body part calculations."""
import numpy as np
import pandas as pd
import matplotlib as plt

def velocity(csv_file, output_path, trial_float):
    """
    Method to calculate velocity of body parts in a video.
    Parameters
        ----------
        csv_file : str
            The path to the csv file containing the body part coordinates.
 
        Returns
        -------
        df : DataFrame
            A pandas DataFrame containing the body part coordinates and velocity.
        """
    
    df = pd.read_csv(csv_file)
    
    # Subtracting the number of title rows in the file
    global frames
    global frame_rate
    frames = len(df) - 1
    frame_rate = frames/trial_float
    timing = 1/frame_rate
    
    # Calculating average velocity in pixels/second (need to input the body parts of interest)
    for column in df[['paw1_x', 'paw1_y', 'paw2_x', 'paw2_y']]:
        df[f'{column}_velocity'] = df[column].diff() / timing
    
    # Calculating overall velocity of a body part
    df['paw1_velocity'] = np.sqrt((df['paw1_x_velocity'] ** 2) + (df['paw1_y_velocity'] ** 2))
    df['paw2_velocity'] = np.sqrt((df['paw2_x_velocity'] ** 2) + (df['paw2_y_velocity'] ** 2))
    
    # Calculating normalized velocity
    df['paw1_velocity'] = df['paw1_velocity'] / df['paw1_velocity'].max()
    df['paw2_velocity'] = df['paw2_velocity'] / df['paw2_velocity'].max()
    
    # Converting the frame indices to time for plotting purposes
    df['x_rescaled'] = df['bodyparts_coords'] / (frame_rate) 
    # Saving the velocity calculations into a csv file for further analysis
    df.to_csv(f'/{output_path}/paw_velocity.csv', index = False)
    
    # Creating graphs for the body parts analyzed
    for column in df[['paw1_velocity', 'paw2_velocity']]:
        plt.figure()
        plt.plot(df['x_rescaled'], df[column])
        plt.ylabel('Normalized velocity')
        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.show()
        
    return df
