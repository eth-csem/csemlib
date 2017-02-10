from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension
module1 = Extension(include_dirs=['./csemlib/models/S20RTS/'], name='s20eval',
                    sources=['./csemlib/models/S20RTS/s20.pyf', './csemlib/models/S20RTS/s20_wrapper.f90',
                             './csemlib/models/S20RTS/sph2v_sub.f'])
setup(
    name='csemlib',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['click', 'numpy', 'scipy', 'matplotlib', 'xarray', 'meshpy', 'numba', 'cython', 'pyvtk', 'boltons', 'PyYAML',
                      'h5py'],
    entry_points='''
    [console_scripts]
    csem=csemlib.csemlib:cli
    ''',
    ext_modules=[module1]
)
