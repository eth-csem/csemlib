import pandas as pd
import numpy as np
import copy

from csemlib.models.one_dimensional import csem_1d_background_eval_point_cloud, csem_1d_background_eval_point_cloud_region_specified
from csemlib.utils import cart2sph, sph2cart, get_rot_matrix, rotate


class GridData:
    """
    Class that serves as a collection point of information on the grid,
    its coordinates and the corresponding data. Data is structured using a pandas DataFrame. Additional functionality
    is build on top of that.
    """

    def __init__(self, x=[], y=[], z=[], components=[], coord_system='cartesian'):
        self.coordinate_system = coord_system
        self.components = []
        if self.coordinate_system == 'cartesian':
            self.coordinates = ['x', 'y', 'z']
        elif self.coordinate_system == 'spherical':
            self.coordinates = ['c', 'l', 'r']
        self.df = pd.DataFrame(np.array((x, y, z)).T, columns=self.coordinates)
        self.add_components(components)

        if self.coordinate_system == 'cartesian':
            self.add_col_lon_rad()
        elif self.coordinate_system == 'spherical':
            self.add_xyz()

    def __getitem__(self, i):
        x, y, z = self.df[self.coordinates].loc[i].values.T
        grid_data = GridData(x, y, z, coord_system=self.coordinate_system)
        for component in self.components:
            grid_data.set_component(component, self.df[component].loc[i].values)
        return grid_data

    def __len__(self):
        return len(self.df)

    def copy(self):
        return copy.deepcopy(self)

    def append(self, griddata):
        self.df = self.df.append(griddata.df)

    def add_components(self, components):
        self.components.extend(components)
        for component in components:
            self.df[component] = np.zeros(len(self.df))

    def del_components(self, components):
        for component in components:
            del self.df[component]
            self.components.remove(component)

    def set_component(self, component, values):
        if component not in self.df.columns:
            self.components.append(component)
        self.df[component] = values

    def get_component(self, component):
        return self.df[component].values

    def get_data(self):
        return self.df[self.components].values

    def get_coordinates(self, coordinate_type=None):
        coordinate_type = coordinate_type or self.coordinate_system

        if coordinate_type == 'spherical':
            return self.df[['c', 'l', 'r']].values
        elif coordinate_type == 'cartesian':
            return self.df[['x', 'y', 'z']].values

    def add_col_lon_rad(self):
        self.df['c'], self.df['l'], self.df['r'] = cart2sph(self.df['x'], self.df['y'], self.df['z'])

    def add_xyz(self):
        self.df['x'], self.df['y'], self.df['z'] = sph2cart(self.df['c'], self.df['l'], self.df['r'])

    def rotate(self, angle, x, y, z):
        rot_mat = get_rot_matrix(angle, x, y, z)
        self.df['x'], self.df['y'], self.df['z'] = rotate(self.df['x'], self.df['y'], self.df['z'], rot_mat)

        # Also update c,l,r coordinates
        self.add_col_lon_rad()

    def add_one_d(self, add_to_components=True):

        one_d_rho, one_d_vpv, one_d_vph, one_d_vsv, one_d_vsh, one_d_eta, one_d_Qmu, one_d_Qkappa = \
            csem_1d_background_eval_point_cloud(self.df['r'])

        self.df['one_d_rho'] = one_d_rho
        self.df['one_d_vpv'] = one_d_vpv
        self.df['one_d_vph'] = one_d_vph
        self.df['one_d_vsv'] = one_d_vsv
        self.df['one_d_vsh'] = one_d_vsh
        self.df['one_d_eta'] = one_d_eta
        self.df['one_d_Qmu'] = one_d_Qmu
        self.df['one_d_Qkappa'] = one_d_Qkappa

        if add_to_components:
            self.set_component('rho', self.df['one_d_rho'])
            self.set_component('vsh', self.df['one_d_vsh'])
            self.set_component('vsv', self.df['one_d_vsv'])
            self.set_component('vph', self.df['one_d_vph'])
            self.set_component('vpv', self.df['one_d_vpv'])
            self.set_component('eta', self.df['one_d_eta'])
            self.set_component('Qmu', self.df['one_d_Qmu'])
            self.set_component('QKappa', self.df['one_d_Qkappa'])

    def del_one_d(self):
        one_d_parameters = ['one_d_rho', 'one_d_vpv', 'one_d_vph', 'one_d_vsv', 'one_d_vsh', 'one_d_eta', 'one_d_Qmu', 'one_d_Qkappa']
        for param in one_d_parameters:
            del self.df[param]

    def add_one_d_discontinuous(self, regions, add_to_components=True, initialize_with_one=False):
        if initialize_with_one:
            one_d_rho = one_d_vpv = one_d_vph = one_d_vsv = one_d_vsh = one_eta = one_Qmu = one_Qkappa\
                = np.ones_like(self.df['r'])
        else:
            one_d_rho, one_d_vpv, one_d_vph, one_d_vsv, one_d_vsh, one_eta, one_Qmu, one_Qkappa = \
                csem_1d_background_eval_point_cloud_region_specified(self.df['r'], regions)


        self.df['one_d_rho'] = one_d_rho
        self.df['one_d_vpv'] = one_d_vpv
        self.df['one_d_vph'] = one_d_vph
        self.df['one_d_vsv'] = one_d_vsv
        self.df['one_d_vsh'] = one_d_vsh
        self.df['one_d_eta'] = one_eta
        self.df['one_d_Qmu'] = one_Qmu
        self.df['one_d_Qkappa'] = one_Qkappa

        if add_to_components:
            self.set_component('rho', self.df['one_d_rho'])
            self.set_component('vsh', self.df['one_d_vsh'])
            self.set_component('vsv', self.df['one_d_vsv'])
            self.set_component('vph', self.df['one_d_vph'])
            self.set_component('vpv', self.df['one_d_vpv'])
            self.set_component('eta', self.df['one_d_eta'])
            self.set_component('Qmu', self.df['one_d_Qmu'])
            self.set_component('QKappa', self.df['one_d_Qkappa'])

    def add_one_d_continuous(self, region_plus_eps, region_min_eps, add_to_components=True, initialize_with_one=False):
        if initialize_with_one:
            one_d_rho = one_d_vpv = one_d_vph = one_d_vsv = one_d_vsh = one_d_eta = one_d_Qmu = one_d_Qkappa\
                = np.ones_like(self.df['r'])
        else:
            # Average the points that lie on the discontinuities
            one_d_rho_peps, one_d_vpv_peps, one_d_vph_peps, one_d_vsv_peps, \
            one_d_vsh_peps, one_d_eta_peps, one_d_Qmu_peps, one_d_Qkappa_peps = \
                csem_1d_background_eval_point_cloud_region_specified(self.df['r'], region_plus_eps)

            one_d_rho, one_d_vpv, one_d_vph, one_d_vsv, \
            one_d_vsh, one_d_eta, one_d_Qmu, one_d_Qkappa = \
                csem_1d_background_eval_point_cloud_region_specified(self.df['r'], region_min_eps)
            one_d_rho = (one_d_rho + one_d_rho_peps) / 2.0
            one_d_vpv = (one_d_vpv + one_d_vpv_peps) / 2.0
            one_d_vph = (one_d_vph + one_d_vph_peps) / 2.0
            one_d_vsv = (one_d_vsv + one_d_vsv_peps) / 2.0
            one_d_vsh = (one_d_vsh + one_d_vsh_peps) / 2.0
            one_d_eta = (one_d_eta + one_d_eta_peps) / 2.0
            one_d_Qmu = (one_d_Qmu + one_d_Qmu_peps) / 2.0
            one_d_Qkappa = (one_d_Qkappa + one_d_Qkappa_peps) / 2.0


        self.df['one_d_rho'] = one_d_rho
        self.df['one_d_vpv'] = one_d_vpv
        self.df['one_d_vph'] = one_d_vph
        self.df['one_d_vsv'] = one_d_vsv
        self.df['one_d_vsh'] = one_d_vsh
        self.df['one_d_eta'] = one_d_eta
        self.df['one_d_Qmu'] = one_d_Qmu
        self.df['one_d_Qkappa'] = one_d_Qkappa

        if add_to_components:
            self.set_component('rho', self.df['one_d_rho'])
            self.set_component('vsh', self.df['one_d_vsh'])
            self.set_component('vsv', self.df['one_d_vsv'])
            self.set_component('vph', self.df['one_d_vph'])
            self.set_component('vpv', self.df['one_d_vpv'])
            self.set_component('eta', self.df['one_d_eta'])
            self.set_component('Qmu', self.df['one_d_Qmu'])
            self.set_component('QKappa', self.df['one_d_Qkappa'])
