import pandas as pd
import numpy as np
from scipy.interpolate import Rbf

'''
UW VF

---
This will need to be run on every machine due to the sheer number of files.
Create a UW folder under images
---
'''

coordinates_df = pd.read_csv("data/UW_coords.csv")
uw_vf = pd.read_csv("data/UW_VF_Data.csv")

cropped = uw_vf.drop(uw_vf.columns[[1]], axis = 1)
cropped = cropped.drop(cropped.columns[np.r_[3:21]], axis = 1)
cropped_df = cropped.drop(cropped.columns[np.r_[57:165]], axis = 1)
cropped_df['Eye'] = cropped_df['Eye'].replace({'Right': 'OD', 'Left': 'OS'})
cropped_np = np.array(cropped_df)

all_coords =  np.array([(coordinates_df["X"][i], -1 * coordinates_df["Y"][i]) for i in range(len(coordinates_df))])

radius = 27
grid_x, grid_y = np.mgrid[-27:27:200j, -27:27:200j]
mask = (np.sqrt(grid_x**2 + grid_y**2) <= radius)
mask_float = mask.astype(np.float32)


"""
For each point in cropped_np, create a visual field map:

1. Take the point’s values (pt[3:]).
2. Interpolate over a grid using RBF with fixed VF coordinates.
3. Apply a circular mask of radius 12.
4. Replace values outside the mask with 0.
5. Save the result (values + mask) as a .npy file named "{pt[0]}_{pt[1]}.npy".

Final grid is (200, 200, 2)
    - [:,:,0]  interpolated visual field values
    - [:,:,1] mask indicating the visual field area (1.0 inside, 0.0 outside)
"""
for pt in cropped_np:
    pt_vf = np.array(pt[3:], dtype=int)

    rbf = Rbf(all_coords[:,0], all_coords[:,1], pt_vf, function='thin_plate')
    grid_z = rbf(grid_x, grid_y)
    masked_grid = np.where(mask, grid_z, np.nan)
    
    values = np.nan_to_num(masked_grid, nan=0.0)
    x = np.stack([values.T, mask_float], axis=-1)
    np.save(f"images/UW/{pt[0]}_{pt[1]}_{pt[2]}.npy", x)