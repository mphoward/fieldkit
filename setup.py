from __future__ import division, absolute_import, print_function

from numpy.distutils.core import Extension
ext = Extension(name='fieldkit._fieldkit',
                sources=['fieldkit/fieldkit.pyf','fieldkit/mt19937.f90','fieldkit/measure.f90','fieldkit/simulate.f90'],
                libraries=['gomp'],
                define_macros=[('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')],
                extra_f90_compile_args=['-fopenmp'])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        name='fieldkit',
        version='0.0.0',
        license='BSD-3-Clause',
        long_description='Field analysis toolkit',
        packages=['fieldkit','fieldkit.test'],
        ext_modules = [ext]
    )
