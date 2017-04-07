import click


@click.group()
def cli():
    pass

@cli.command()
@click.option('--filename', prompt='Enter filename',
              help='Salvus exodus file.')
@click.option('--verbose/--quiet', default=True)
def add_discontinous_csem_salvus(filename, verbose):
    """ Adds discontinous csem to a salvus mesh file"""
    from .salvus import add_csem_to_discontinuous_exodus

    add_csem_to_discontinuous_exodus(filename=filename, eval_crust=True, eval_s20=True, eval_europe=True,
                                     eval_japan=True, eval_australia=True, eval_south_atlantic=True)


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