""" Measure properties of fields and domains.
"""
from __future__ import division
import numpy as np
import skimage.measure
from fieldkit.mesh import TriangulatedSurface

def volume(field, threshold, N, seed=None):
    """ Compute the volume for a domain.

    Perform Monte Carlo integration in the periodic cell to determine
    the volume having `field` exceed `threshold`.

    Parameters
    ----------
    field : :py:class:`~fieldkit.mesh.Field`
        The field to analyze.
    threshold : float
        Threshold tolerance for the field to consider a lattice site "filled".
    N : int
        Number of samples to take.
    seed : int or None
        Seed to the NumPy random number generator. If `None`, the random
        seed is not modified.

    Returns
    -------
    float
        The volume of the periodic cell where `field` exceeds the `threshold`.

    Notes
    -----
    The volume is sampled by generating `N` random tuples for the fractional
    coordinates in the periodic cell. The value of `field` at these points
    is determined by linear interpolation. The domain volume fraction
    is estimated as the number of samples having `field` exceed `threshold`
    divided by `N`, which is multiplied by the cell volume to give the domain
    volume.

    """
    # interpolator for the field
    f = field.interpolator()

    # Monte Carlo sampling of the interpolated field
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.uniform(low=0.0, high=1.0, size=(int(N),3))
    hits = np.sum(f(samples) >= threshold)

    return (hits/N) * field.mesh.lattice.volume

def triangulate(field, threshold):
    """ Triangulate the surface of a domain using the Marching Cubes algorithm.

    Parameters
    ----------
    field : :py:class:`~fieldkit.mesh.Field`
        The field to triangulate.
    threshold : float
        Threshold tolerance for the field to consider a lattice site "filled".

    Returns
    -------
    :py:class:`~fieldkit.mesh.TriangulatedSurface`
        Triangulated surface.

    """
    # perform marching cubes on the fractional lattice
    verts,faces,normals,_ = skimage.measure.marching_cubes_lewiner(field.buffered(), level=threshold, spacing=field.mesh.step/field.mesh.lattice.L)

    # map the vertices and normals into the triclinic cell
    verts = field.mesh.lattice.as_coordinate(verts)
    normals = field.mesh.lattice.as_coordinate(normals)

    # generate triangulated surface
    surface = TriangulatedSurface()
    surface.add_vertex(verts, normals)
    surface.add_face(faces)

    return surface

def surface_area(surface):
    """ Compute the surface of a triangulated mesh.

    Parameters
    ----------
    surface : :py:class:`~fieldkit.mesh.TriangulatedSurface`
        Triangulated mesh to evaluate.

    Todo
    -----
    This method needs to be tested in periodic boundary conditions to see if it works correctly.

    """
    return skimage.measure.mesh_surface_area(surface.vertex, np.asarray(surface.face))
