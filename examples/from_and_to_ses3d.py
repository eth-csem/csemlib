from csemlib.background.grid_data import GridData
from csemlib.models.crust import Crust
from csemlib.models.s20rts import S20rts
from csemlib.models.ses3d import Ses3d
from csemlib.io.readers import read_from_ses3d_block
from csemlib.io.writers import write_to_ses3d


x, y, z = read_from_ses3d_block('/Users/Andreas/CSEM/csemlib/regional_models/japan/')
grid_data = GridData(x, y, z)

# Initialise with CSEM 1D background model.
grid_data.add_one_d()
grid_data.set_component('vsv', grid_data.df['one_d_vsv'])

# Add s20rts
s20 = S20rts()
s20.eval_point_cloud_griddata(grid_data)

# Add Crust
cst = Crust()
cst.eval_point_cloud_grid_data(grid_data)

# Add Japan
ses3d = Ses3d('/Users/Andreas/CSEM/csemlib/regional_models/japan', grid_data.components)
ses3d.eval_point_cloud_griddata(grid_data)

# Generate output in ses3d format.
write_to_ses3d('/Users/Andreas/CSEM/csemlib/regional_models/japan','vsv','vsv_new',grid_data)
write_to_ses3d('/Users/Andreas/CSEM/csemlib/regional_models/japan', 'vsh', 'vsh_new', grid_data)
write_to_ses3d('/Users/Andreas/CSEM/csemlib/regional_models/japan', 'vpv', 'vpv_new', grid_data)
write_to_ses3d('/Users/Andreas/CSEM/csemlib/regional_models/japan', 'vph', 'vph_new', grid_data)
write_to_ses3d('/Users/Andreas/CSEM/csemlib/regional_models/japan', 'rho', 'rho_new', grid_data)
