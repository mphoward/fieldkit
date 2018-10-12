""" Identify and characterize domains.
"""
from __future__ import division
import numpy as np
import networkx

def find(field, threshold):
    """ Finds connected domains in a field.

    A domain is defined to be a connected region of lattice
    points having a `field` value greater than or equal to
    `threshold`, subject to periodic boundary conditions of the
    lattice.

    Parameters
    ----------
    field : :py:obj:`~membranekit.mesh.Field`
        The field to analyze for continuous domains.
    threshold : float
        Threshold tolerance for the field to consider a lattice
        site "filled".

    Returns
    -------
    iterator
        An iterator for indexes in :py:obj:`~membranekit.mesh.Mesh`.
        Two nested iterators are returned: the first one is over
        domains, while the second is over the points in each domain.

    Notes
    -----
    The connected domains are determined using a graph-based approach,
    which requires the `networkx` package. Points meeting the threshold
    tolerance are considered as nodes in the graph. Edges are defined
    between nodes that are adjacent in the lattice. The domains are
    identified within the cell by finding the connected components of
    the graph. Performance is generally good, but the algorithm may
    struggle for large numbers of nodes or domains.

    """
    flags = field.field >= threshold
    g = networkx.Graph()
    for pt in np.ndindex(field.shape):
        if flags[pt]:
            g.add_node(pt)
            for neigh in field.mesh.neighbors(pt,full=False):
                if flags[neigh]:
                    g.add_edge(pt, neigh)

    # domains will be the connected components of the graph
    return networkx.connected_components(g)

def volume(field, threshold, N, seed=None):
    """ Compute the volume for a domain.

    Perform Monte Carlo integration in the periodic cell to determine
    the volume having `field` exceed `threshold`.

    Parameters
    ----------
    field : :py:obj:`~membranekit.mesh.Field`
        The field to analyze for continuous domains.
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
    samples = np.random.uniform(low=0.0, high=1.0, size=(N,3))
    hits = np.sum(f(samples) >= threshold)

    return (hits/N) * field.mesh.lattice.volume
