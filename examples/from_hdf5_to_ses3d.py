import os
import io
import yaml

from csemlib.models.ses3d import Ses3d
from csemlib.io.writers import write_to_ses3d


csemlib_directory, _ = os.path.split(os.path.split(__file__)[0])
models_directory = os.path.join(csemlib_directory, 'regional_models')
model_dir = os.path.join(models_directory, 'australia_2010')

# Read yaml file containing information on the ses3d submodel.
with io.open(os.path.join(model_dir, 'modelinfo.yml'), 'rt') as fh:
    model_info = yaml.load(fh)
components = model_info['components']

ses3d = Ses3d(model_dir, components)
ses3d.init_grid_data_hdf5()
grid_data = ses3d.grid_data_ses3d


# Generate output in ses3d format.\\
for component in components:
    write_to_ses3d(model_dir, component, '{}_new'.format(component), grid_data)