import pandas as pd
from pathlib import Path
import os

TEST_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
RESOURCES_DIR = TEST_DIR / "resources"

trial_coords = RESOURCES_DIR / "corr_test.csv"
adj_coords = RESOURCES_DIR / "adjusted_coords.csv"

def trial_coordinates(trial_coords):
    
    df = pd.read_csv(trial_coords, header=None)

    return df

def velocity_coordinates(adj_coords):
        
    df = pd.read_csv(adj_coords)
        
    return df