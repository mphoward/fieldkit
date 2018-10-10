""" Identify and characterize domains.
"""
import networkx
import numpy as np

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
