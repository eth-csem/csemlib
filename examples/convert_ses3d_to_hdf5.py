import os

from csemlib.models.ses3d import Ses3d

csemlib_directory, _ = os.path.split(os.path.split(__file__)[0])
model_directory = os.path.join(csemlib_directory, 'regional_models')

ses3d = Ses3d(os.path.join(model_directory, 'australia_2010'), ['rho', 'vsh', 'vsv', 'vph', 'vpv'])
ses3d.read()
# Write to hdf5 with default filename
ses3d.write_to_hdf5()
