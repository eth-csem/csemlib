import click


@click.group()
def cli():
    pass

#===============================================================================
#- Evaluate CSEM on a Fibonacci sphere and write to file. ----------------------
#===============================================================================

@cli.command()
@click.option('--depth', prompt='Enter depth in km',
              help='Depth in km, defaults to 150',
              default=150.0)
@click.option('--grid_spacing', prompt='Enter grid spacing',
              help='Grid spacing in km, defaults to 200',
              default=200.0)
@click.option('--parameter', prompt="Enter parameter",
              help="Name of parameter. Choose vsv, vsh, vpv, etc., "
                   "defaults to vsv.",
              default="vsv")
@click.option('--filename', help='VTK filename (optional)', default=None)
def write_csem_depth_slice(depth, grid_spacing, parameter, filename):
    """ Writes a CSEM depth slice to vtk"""

    from csemlib.api import csem2vtk
    csem2vtk(depth, grid_spacing, parameter, filename)
