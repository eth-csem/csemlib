from evaluate_csem import evaluate_csem
import numpy as np
from csemlib.utils import sph2cart, lat2colat
import pandas as pd


# Generate point cloud in list form
lats = np.linspace(40, 89, 99)
lons = np.linspace(-100, 40, 281)
depths = np.linspace(0, 600, 61)
rads = 6371.0 - depths

all_lats, all_lons, all_rads = np.meshgrid(lats, lons, rads)
all_lats, all_lons, all_rads = np.array(
        (all_lats.ravel(), all_lons.ravel(), all_rads.ravel())
    )

# Convert to CSEM coordinates
all_colats = lat2colat(all_lats)
all_colats_rad = np.deg2rad(all_colats)
all_lons_rad = np.deg2rad(all_lons)
x, y, z = sph2cart(all_colats_rad, all_lons_rad, all_rads)

# Evaluate grid points
grid_data = evaluate_csem(x,y,z)

# Write gridpoints to CSV
df = pd.DataFrame({'lats': all_lats, 'lons': all_lons, "depths": 6371.0 - all_rads,
            "VSV":grid_data.df["vsv"].values,
            "VSH":grid_data.df["vsh"].values,
            "VPV":grid_data.df["vpv"].values,
            "VPH":grid_data.df["vph"].values,
            "RHO":grid_data.df["rho"].values,
            "ETA":grid_data.df["eta"].values,
              })
df.to_csv("lat_lon_depth", index=False)