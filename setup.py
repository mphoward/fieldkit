from __future__ import division, absolute_import, print_function

from numpy.distutils.core import Extension
exts = [Extension(name='fieldkit._measure', sources=['fieldkit/measure.pyf','fieldkit/measure.f90'], define_macros=[('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')]),
        Extension(name='fieldkit._simulate', sources=['fieldkit/simulate.pyf','fieldkit/simulate.f90'], define_macros=[('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')])]

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        name='fieldkit',
        version='0.0.0',
        packages=['fieldkit','fieldkit.test'],
        license='BSD-3-Clause',
        long_description='Field analysis toolkit',
        ext_modules = exts
    )
