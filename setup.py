import os

from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import inspect

src = './csemlib/models/S20RTS/'
module1 = Extension('s20eval', include_dirs=[src],
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
    package_data={
        "csemlib":
            [os.path.join("lib", "s20eval.so")]},
    include_package_data=True,
    install_requires=['click', 'numpy', 'scipy', 'matplotlib', 'xarray', 'meshpy', 'numba', 'cython', 'pyvtk', 'boltons', 'PyYAML',
                      'h5py'],
    entry_points='''
    [console_scripts]
    csem=csemlib.csemlib:cli
    ''',
    ext_package='s20eval.lib',
    ext_modules=[module1]
)