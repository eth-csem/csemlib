import click


@click.group()
def cli():
    pass

regions_dict=dict(eval_crust=True, eval_s20=True, eval_south_atlantic=True, eval_australia=True,
                  eval_japan=True, eval_europe=True, eval_marmara_2017=True,
                  eval_south_east_asia_2017=True, eval_iberia_2015=True, eval_iberia_2017=True,
                  eval_north_atlantic_2013=True, eval_north_america=True, eval_global_1=True)

#=======================================================================================================================
#- Add values to a continuous Salvus mesh. -----------------------------------------------------------------------------
#=======================================================================================================================

@cli.command()
@click.option('--filename',
              help="Salvus continuous exodus file.", required=True)
@click.option('--with_topography',
              help="Account for topography/ellipticity, requires 1D radius in in mesh",
              is_flag=True)
def add_continuous_csem_salvus(filename, with_topography):
    """ Adds CSEM to a continuous salvus mesh file"""
    from .salvus import add_csem_to_continuous_exodus

    add_csem_to_continuous_exodus(filename=filename, regions_dict=regions_dict,
                                  with_topography=with_topography)

@cli.command()
@click.option('--filename',
              help="Salvus continuous exodus file.", required=True)
@click.option('--with_topography',
              help="Account for topography/ellipticity, requires 1D radius in in mesh",
              is_flag=True)
def add_s20_to_isotropic_mesh(filename, with_topography):
    """ Adds CSEM to a continuous salvus mesh file"""
    from .salvus import add_s20_to_isotropic_exodus

    add_s20_to_isotropic_exodus(filename=filename, regions_dict=regions_dict,
                                  with_topography=with_topography)

#=======================================================================================================================
#- Add values to a discontinuous Salvus mesh. --------------------------------------------------------------------------
#=======================================================================================================================

@cli.command()
@click.option('--filename', prompt='Enter filename', required=True,
              help='Salvus discontinuous exodus file.')
def add_discontinuous_csem_salvus(filename):
    """ Adds CSEM to a discontinuous salvus mesh file"""
    from .salvus import add_csem_to_discontinuous_exodus

    add_csem_to_discontinuous_exodus(filename=filename, regions_dict=regions_dict)

#=======================================================================================================================
#- Evaluate CSEM on a Fibonacci sphere and write to file. --------------------------------------------------------------
#=======================================================================================================================

#- COMMENT: Maybe one should also add here the option of not having all submodels. Alternatively, this option could be
#- eliminated all together.

@cli.command()
@click.option('--depth', prompt='Enter depth in km', help='Depth in km.', default=150.0)
@click.option('--resolution', prompt='Enter resolution', help='resolution', default=200.0)
@click.option('--filename', help='vtk filename', default=None)

def write_csem_depth_slice(depth, resolution, filename):
    """ Writes a CSEM depth slice to vtk"""

    from .assemble_csem import depth_slice_to_vtk
    depth_slice_to_vtk(depth, resolution, filename)


#=======================================================================================================================
#- Add values to a continuous SalvusV2 mesh. -----------------------------------------------------------------------------
#=======================================================================================================================

@cli.command()
@click.option('--filename',
              help="Salvus continuous exodus file.", required=True)
def add_continuous_csem_salvusv2(filename):
    """ Adds CSEM to a continuous salvusv2 mesh file"""
    from .salvus import add_csem_to_salvusv2

    add_csem_to_salvusv2(filename=filename, regions_dict=regions_dict)