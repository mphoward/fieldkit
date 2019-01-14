# fieldkit

fieldkit is a Python package for analyzing and manipulating 3D data on a mesh.
It simplifies the tasks of reading field data on a grid and characterizing
it. Example analysis tools include domain identification, percolation detection,
surface triangulation, and computing advanced geometric and topological measures.
Its inputs are flexible, making it well-suited to analyzing field data collected by
a variety of simulation or experimental techniques.

## Requirements

The following software is required to use fieldkit:

* Python 2 or 3
* [NumPy] and [SciPy]
* [scikit-image]
* [NetworkX]
* A Fortran 90 compiler (tested: gfortran)

The easiest way to ensure all of these are installed is with a Python package
manager such as [Anaconda].

## Installation

Once all dependencies are met, installation can be as simple as:
```
python setup.py install
```
Other installation methods may be made available in future.

You can verify your installation using the included suite of unit tests:
```
python -c "import fieldkit; fieldkit.test.run()"
```

## Acknowledgments

This work was supported as part of the Center for Materials for Water and
Energy Systems ([M-WET]), an Energy Frontier Research Center funded by the
U.S. Department of Energy, Office of Science, Basic Energy Sciences under
Award #DE-SC0019272.

fieldkit was inspired by basic analysis tools shared by the
[Fredrickson Research Group]. In addition, the authors of the following codes
are gratefully acknowledged:

* `fieldkit.io.read_polyfts` was shared by [Josh Lequieu].
* `fieldkit.measure.minkowski` uses Fortran subroutines written by
  [Michielsen and De Raedt] as part of their review article on Minkowski
  functionals in "Integral-geometry morphological image analysis",
  *Phys. Reports* **347**, 461 (2001).

[NumPy]: https://www.numpy.org
[SciPy]: https://www.scipy.org
[scikit-image]: https://scikit-image.org
[Networkx]: https://networkx.github.io
[Anaconda]: https://www.anaconda.com
[M-WET]: https://mwet.utexas.edu
[Fredrickson Research Group]: https://www.mrl.ucsb.edu/~fredrickson
[Josh Lequieu]: https://www.joshlequieu.com
[Michielsen and De Raedt]: https://doi.org/10.1016/S0370-1573(00)00106-X
