from csemlib.background.grid_data import GridData
from csemlib.models.specfem import TriLinearInterpolator, Specfem
import numpy as np
from csemlib.helpers import load_lib
lib = load_lib()

specfem = Specfem(interp_method='trilinear_interpolation')

x, y, z = specfem.nodes.T
# x = x[:5]
# y = y[:5]
# z = z[:5]


grid_data = GridData(x,y,z)
grid_data.set_component('vsv', np.ones(len(grid_data)))
grid_data.set_component('vsh', np.ones(len(grid_data)))

specfem.eval_point_cloud_griddata(grid_data)

vsv = grid_data.get_component('vsv')

print((vsv))

print(' Finished messing around')
tlp = TriLinearInterpolator()

pnt = np.array((x,y,z)).T
vtx = np.array([[3608.680926, -3521.022229, 3790.311898],
                [3549.003838, -3549.003838, 3820.440184],
                [3560.261266, -3473.628629, 3832.557725],
                [3619.683547, -3445.819043, 3801.875017],
                [3624.888660, -3536.841487, 3807.341391],
                [3652.347745, -3476.916049, 3836.183115],
                [3592.390409, -3504.973607, 3867.139617],
                [3564.950310, -3564.950310, 3837.603726],
                ])

vtx = np.array(vtx)
pnt = np.array([3608.680926, -3521.022229, 3790.311898])

#print(vtx)
solution = (tlp.check_hull(pnt, vtx))

#print(tlp.interpolate_at_point(solution[1]))

# vtx = np.array([
#     [-2, -2, -2],
#     [-2, +2, -2],
#     [+2, +2, -2],
#     [+2, -2, -2],
#     [-2, -2, +2],
#     [+2, -2, +2],
#     [+2, +2, +2],
#     [-2, +2, +2]])
# pnt = np.array([0,0,0])
# a = tlp.inverse_coordinate_transform(pnt, vtx)
# print(a)




# -2, -2, -2, -2, +2, -2, +2, +2, -2, +2, -2, -2, -2, -2, +2, +2, -2,
#         +2, +2, +2, +2, -2, +2, +2;


#
# specfem = Specfem(interp_method='trilinear_interpolation')
# first_element = specfem.connectivity[0,:]
# element_vtcs = specfem.nodes[first_element]
#
# permutation = [0,3,2,1,4,5,6,7] # looks like this is the correct one
# i = np.argsort(permutation)
# element_perturbed = first_element[i]
# element_vtcs = specfem.nodes[element_perturbed]
#
# #pnt = np.mean(element_vtcs, axis=0)
# pnt = (element_vtcs[4,:] + element_vtcs[6,:]) / 2
# solution = tlp.check_hull(pnt, element_vtcs)
# print(solution)
# print(tlp.interpolate_at_point(solution[1]))
#
import pyvtk
from pyvtk import PointData, Scalars
vtkElements = pyvtk.VtkData(pyvtk.UnstructuredGrid(
    specfem.nodes, hexahedron=specfem.connectivity),
    PointData(Scalars(grid_data.get_component('vsv'), 'node_number')),
               "Mesh")

vtkElements.tofile('specfem_mesh.vtk')

