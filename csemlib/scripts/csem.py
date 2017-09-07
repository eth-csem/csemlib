import click


@click.group()
def cli():
    pass

@cli.command()
@click.option('--filename',
              help='Salvus continuous exodus file.')
@click.option('--verbose/--quiet', default=True)
def add_continuous_csem_salvus(filename, verbose):
    """ Adds discontinuous csem to a salvus mesh file"""
    from .salvus import add_csem_to_continuous_exodus

    add_csem_to_continuous_exodus(filename=filename, eval_crust=True, eval_s20=True, eval_south_atlantic=True,
                                  eval_australia=True, eval_japan=True, eval_europe=True, eval_marmara_2017=True,
                                  eval_south_east_asia_2017=True, eval_iberia_2015=True, eval_iberia_2017=True,
                                  eval_north_atlantic_2013=True, eval_north_america=True)

@cli.command()
@click.option('--filename', prompt='Enter filename',
              help='Salvus discontinuous exodus file.')
@click.option('--verbose/--quiet', default=True)
def add_discontinuous_csem_salvus(filename, verbose):
    """ Adds discontinuous csem to a salvus mesh file"""
    from .salvus import add_csem_to_discontinuous_exodus

    add_csem_to_discontinuous_exodus(filename=filename, eval_crust=True, eval_s20=True, eval_south_atlantic=True,
                                     eval_australia=True, eval_japan=True, eval_europe=True, eval_marmara_2017=True,
                                     eval_south_east_asia_2017=True, eval_iberia_2015=True, eval_iberia_2017=True,
                                     eval_north_atlantic_2013=True, eval_north_america=True)

@cli.command()
@click.option('--depth', prompt='Enter depth in km',
              help='Depth in kom.', default=150.0)
@click.option('--resolution', prompt='Enter resolution',
              help='resolution', default=200.0)
@click.option('--filename',
              help='vtk filename', default=None)
def write_csem_depth_slice(depth, resolution, filename):
    """ Writes a CSEM depth slice to vtk"""

    from .assemble_csem import depth_slice_to_vtk
    depth_slice_to_vtk(depth, resolution, filename)