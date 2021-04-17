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
@click.option('--resolution', prompt='Enter resolution',
              help='Resolution in km, defaults to 200',
              default=200.0)
@click.option('--parameter', prompt="Enter parameter",
              help="Name of parameter. Choose vsv, vsh, vpv, etc., "
                   "defaults to vsv.",
              default="vsv")
@click.option('--filename', help='VTK filename (optional)', default=None)
def write_csem_depth_slice(depth, resolution, parameter, filename):
    """ Writes a CSEM depth slice to vtk"""

    from csemlib import api
    api.depth_slice_to_vtk(depth, resolution, parameter, filename)
