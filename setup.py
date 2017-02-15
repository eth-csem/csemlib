import os

from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import inspect

src = 'csemlib/src/'

module1 = Extension('s20eval',
                    sources=[
                        os.path.join(src, 's20.pyf'),
                        os.path.join(src, 's20_wrapper.f90'),
                        os.path.join(src, 'sph2v_sub.f')])

def readme():
    with open('README.rst') as f:
        return f.read()



setup(
    name='csemlib',
    version='0.1',
    long_description=readme(),
    packages=find_packages(),
    include_package_data=True,
    install_requires=['click', 'numpy', 'scipy', 'matplotlib', 'xarray', 'meshpy', 'numba', 'cython', 'pyvtk', 'boltons', 'PyYAML',
                      'h5py'],
    entry_points='''
    [console_scripts]
    csem=csemlib.csemlib:cli
    ''',
    ext_package='csemlib.lib',
    ext_modules=[module1]
)