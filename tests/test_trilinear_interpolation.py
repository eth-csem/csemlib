from csemlib.models.specfem import TriLinearInterpolator, Specfem
import numpy as np

tlp = TriLinearInterpolator()

pnt = np.array([-2, -2.0, 0])

vtx = np.array([-2, -2, -2, -2, +2, -2, +2, +2, -2, +2, -2, -2, -2, -2, +2, +2, -2, +2, +2, +2, +2, -2, +2, +2])

specfem = Specfem()
first_element = specfem.connectivity[0,:]
element_vtcs = specfem.nodes[first_element]

your_permutation = [0,3,1,2,4,5,7,6]
i = np.argsort(your_permutation)

element_perturbed = first_element[i]

element_vtcs = specfem.nodes[element_perturbed]

pnt = element_vtcs[3, :] / 2 + element_vtcs[5, :] / 2
print(element_vtcs)
print(pnt)
#vtx = np.array([-2, +2, -2, +2, +2, -2, +2, -2, -2, -2, -2, +2, +2, -2, +2, +2, +2, +2, -2, +2, +2, -2, -2, -2])


solution = tlp.check_hull(pnt, element_vtcs)
print(solution)
print(tlp.interpolate_at_point(solution[1]))

#
# #vtx = np.reshape(vtx, (8, 3set nocompatible              " required                                                 ate_transform(pnt, vtx)
#
# element_1 = specfem.connectivity[0, :]
# element_1 = element_1[i]
#
#
# vortex = specfem.nodes[element_1]
#


# import pyvtk
# from pyvtk import PointData, Scalars
# vtkElements = pyvtk.VtkData(pyvtk.UnstructuredGrid(
#     specfem.nodes, hexahedron=element_1),
#     PointData(Scalars(np.arange(len(specfem.nodes)), 'node_number')),
#                "Mesh")
#
# vtkElements.tofile('specfem_mesh.vtk')

