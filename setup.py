import os

from setuptools import find_packages
from numpy.distutils.core import setup, Extension


src = os.path.join('csemlib', 'src')


module1 = Extension('s20eval',
                    sources=[
                        os.path.join(src, 's20.pyf'),
                        os.path.join(src, 's20_wrapper.f90'),
                        os.path.join(src, 'sph2v_sub.f')])

lib = Extension('csemlib',
                sources=[
                    os.path.join(src, "s20_gridded.c"),
                    os.path.join(src, "add_crust.c"),
                    os.path.join(src, "centroid.c"),
                    os.path.join(src, "trilinearinterpolator.c")],
                extra_compile_args=["-O3"])


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='csemlib',
    version='0.1',
    long_description=readme(),
    packages=find_packages(),
    include_package_data=True,
    dependency_links=['https://github.com/eth-csem/pyexodus/archive/master.zip#egg=pyexodus-master'],
    install_requires=['click', 'numpy', 'scipy', 'matplotlib', 'xarray', 'meshpy', 'numba', 'cython', 'pyvtk', 'boltons', 'PyYAML',
                      'h5py', 'pyexodus'],
    entry_points='''
    [console_scripts]
    csem=csemlib.scripts.csem:cli
    ''',
    ext_package='csemlib.lib',
    ext_modules=[module1, lib]
)
