# from csemlib.models.specfem import TriLinearInterpolator, Specfem
# import numpy as np
#
# tlp = TriLinearInterpolator()
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
# import pyvtk
# from pyvtk import PointData, Scalars
# vtkElements = pyvtk.VtkData(pyvtk.UnstructuredGrid(
#     specfem.nodes, hexahedron=element_1),
#     PointData(Scalars(np.arange(len(specfem.nodes)), 'node_number')),
#                "Mesh")
#
# vtkElements.tofile('specfem_mesh.vtk')

