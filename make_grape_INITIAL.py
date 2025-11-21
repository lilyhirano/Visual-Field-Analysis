import pandas as pd
import numpy as np
from scipy.interpolate import Rbf

'''
GRAPE Initial VF
'''

grape_vf = pd.read_csv("data/GRAPE_VF.csv")
cropped_df = grape_vf.drop(columns = ['Age', 'Gender', 'IOP', 'CCT', 'Total Visits', 'PLR2', 'PLR3', 'MD', 'Mean', 'S', 'N','I', 'T', 'Corresponding CFP', 'Category of Glaucoma'])

cropped_np = np.array(cropped_df)


all_coords = np.array([
    (0,0), (1,-1), (-1,-1), (-1,1), (1,1), (2,-2), (-2,-2), (-2,2),
    (2,2), (4,-1), (4,-4), (1,-4), (-1,-4), (-4,-4), (-4,-1), (-4,1), (-4,4),
    (-1,4), (1,4), (4,4), (4,1), (6,0), (6,-6), (2,-6), (-2,-6), (-6,-6),
    (-6,-1), (-6,1), (-6,6), (-2,6), (2,6), (6,6), (8,-1), (8,-6), (8,-8), (6,-8), (2,-8), 
    (-2,-8), (-6,-8), (-8,-8), (-8,-6), (-8,-1), (-8,1), (-8,6), (-8,8), (-6,8), (-2,8), (2,8), (6,8),
    (8,8), (8,6), (8,1), (10,-3), (3,-10), (-3,-10), (-10, -2), (-10,2), (-3,10), (3,10), (10,3)
])

radius = 12

grid_x, grid_y = np.mgrid[-12:12:200j, -12:12:200j]
mask = (np.sqrt(grid_x**2 + grid_y**2) <= radius)
mask_float = mask.astype(np.float32)


"""
For each point in cropped_np, create a visual field map:

1. Take the point’s values (pt[2:]).
2. Interpolate over a grid using RBF with fixed VF coordinates.
3. Apply a circular mask of radius 12.
4. Replace values outside the mask with 0.
5. Save the result (values + mask) as a .npy file named "{pt[0]}_{pt[1]}.npy".

Final grid is (200, 200, 2)
    - [:,:,0]  interpolated visual field values
    - [:,:,1] mask indicating the visual field area (1.0 inside, 0.0 outside)
"""
for pt in cropped_np:
    pt_vf = np.array(pt[2:], dtype=int)

    rbf = Rbf(all_coords[:,0], all_coords[:,1], pt_vf, function='thin_plate')
    grid_z = rbf(grid_x, grid_y)
    masked_grid = np.where(mask, grid_z, np.nan)
    
    values = np.nan_to_num(masked_grid, nan=0.0)
    x = np.stack([values.T, mask_float], axis=-1)
    np.save(f"images/GRAPE/Initial/{pt[0]}_{pt[1]}.npy", x)