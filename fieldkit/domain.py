""" Identify and characterize domains.
"""
from __future__ import division
import numpy as np
import networkx
from fieldkit.mesh import Domain

def digitize(field, threshold):
    """ Digitize a :py:class:`~fieldkit.mesh.Field`.

    A scalar :py:class:`~fieldkit.mesh.Field` is converted
    into a set of nodes using a `threshold` tolerance. Nodes
    having `field >= threshold` are included in the digitized
    domain, while other nodes are neglected.

    Parameters
    ----------
    field : :py:class:`~fieldkit.mesh.Field`
        The field to digitize.
    threshold : float
        The threshold tolerance for digitizing the field.

    Returns
    -------
    :py:class:`~fieldkit.mesh.Domain`
        The domain corresponding to the nodes having
        `field >= threshold`.

    """
    flags = field.field >= threshold
    nodes = field.mesh.indices[flags]
    return Domain(field.mesh, nodes)

def find(domain):
    """ Finds connected domains within a domain.

    A domain is defined to be a connected region of lattice
    points, subject to periodic boundary conditions.

    Parameters
    ----------
    domain : :py:class:`~fieldkit.mesh.Domain`
        The set of nodes to seek connected domains in.

    Returns
    -------
    tuple
        A tuple of all :py:class:`~fieldkit.mesh.Domain` objects
        identified within the `domain`. At most, there is only
        one domain returned, but many can be identified if the points
        in the `domain` are highly disconnected.

    Notes
    -----
    The connected domains are determined using a graph-based approach,
    which requires the `networkx` package. Performance is generally good,
    but the algorithm may struggle for large numbers of nodes or domains.

    """
    comps = networkx.connected_components(domain.graph)
    return tuple([Domain(domain.mesh,list(c)) for c in comps])

def burn(domain):
    """ Compute the burn number distribution and medial axis for a domain.

    Implements the burning algorithm + medial axis protocol outlined by
    Lindquist et al. [1]_ A burning algorithm ignites a "flame" from the
    boundaries of the domain, and marches the flame front through. The medial
    axis is defined by the locus of voxels where the flame front is extinguished
    by an opposing front.

    Parameters
    ----------
    domain : :py:class:`~fieldkit.mesh.Domain`
        The set of nodes (domain) to burn.

    Returns
    -------
    burn_number : numpy.ndarray
        The burn number for each voxel in the domain. A burn number of 0 means
        a boundary, and all other vales indicate the iteration the burn reached
        the voxel.
    axis : numpy.ndarray
        An `N`x3 tuple of integers giving the indexes of the mesh points identified
        as the medial axis. These can be converted to coordinates using the
        corresponding :py:class:`~fieldkit.mesh.Mesh` for `field`.

    References
    ----------
    .. [1] W. B. Lindquist et al. "Medial axis analysis of void structure in
       three-dimensional tomographic images of porous media", J. Geophys. Res.
       101 (B4), 8297--8310.

    """
    BURN_SIGNAL = -1
    burn_number = np.zeros(domain.mesh.shape, dtype=np.int32)
    burn_number[domain.mask] = BURN_SIGNAL

    medial_axis = np.zeros(domain.mesh.shape, dtype=bool)

    # iterate through while some burn numbers have not been determined
    k = 1
    while np.any(burn_number == BURN_SIGNAL):
        burn_vecs = {}
        for pt in np.ndindex(burn_number.shape):
            # only work on unchecked points
            if burn_number[pt] != BURN_SIGNAL:
                continue

            # get neighboring points and burn vectors
            neighs = domain.mesh.neighbors(pt, full=True)
            burn_vecs[pt] = []
            for n in neighs:
                if burn_number[n] == k-1:
                    # this is minimum image on the mesh
                    dn = np.array(pt) - np.array(n)
                    dn -= domain.mesh.shape * np.round(dn.astype(float)/domain.mesh.shape).astype(int)
                    burn_vecs[pt].append(dn)

            # this point gets burned if any vectors are coming into it
            if len(burn_vecs[pt]) > 0:
                burn_number[pt] = k
            else:
                continue

            # Fig. 2a: check for medial axis condition based on shared burning
            if len(burn_vecs[pt]) > 1:
                for i,vi in enumerate(burn_vecs[pt]):
                    for vj in burn_vecs[pt][i+1:]:
                        if np.all(vi + vj == 0):
                            medial_axis[pt] = True

            # Fig. 2c: handle case where burn extinguishes between two voxels
            if not medial_axis[pt]:
                for v in burn_vecs[pt]:
                    # follow the burn vector to the next neighbor with pbcs (no self images)
                    n = tuple((pt + v) % domain.mesh.shape)
                    if n == pt:
                        continue

                    # look at burning of this neighbor if at same iteration
                    # if burn is also coming the opposite direction, then both are medial axis
                    if burn_number[n] == k and n in burn_vecs:
                        neg_v = tuple(-v)
                        for nv in burn_vecs[n]:
                            if neg_v == tuple(nv):
                                medial_axis[n] = True
                                medial_axis[pt] = True
                                break

            # Fig. 2d: perpendicular hit
            if not medial_axis[pt] and len(burn_vecs[pt]) > 1:
                for i,vi in enumerate(burn_vecs[pt]):
                    for vj in burn_vecs[pt][i+1:]:
                        if np.dot(vi,vj) == 0:
                            v1 = tuple((pt + vi - vj) % domain.mesh.shape)
                            v2 = tuple((pt - vi + vj) % domain.mesh.shape)
                            if burn_number[v1] == k-1 and burn_number[v2] == k-1:
                                medial_axis[pt] = True
        k += 1


    # extract the indices corresponding to the medial axis
    axis = domain.mesh.indices[medial_axis]

    return burn_number, axis

def is_connected(domain):
    """ Test if a :py:class:`~fieldkit.mesh.Domain` is connected.

    Returns
    -------
    bool
        `True` if the `domain` is a single connected component, and
        `False` otherwise.

    """
    return networkx.is_connected(domain.graph)

def is_percolated(domain, axis):
    """ Test if a :py:class:`~fieldkit.mesh.Domain` is percolated.

    A `domain` is defined to be percolated along an `axis` if there
    exists a continuous path spanning the underlying mesh in the
    direction of `axis`. In practice, this is checked by unwrapping
    the graph corresponding to the `domain` and searching for a path
    from any node on one edge of `axis` to its unwrapped image on the
    other edge. A `domain` may accordingly be percolated in 1, 2, or
    3 dimensions depending on the geometry, and each `axis` should
    usually be checked sequentially.

    Returns
    -------
    bool
        `True` if the `domain` is percolated through the periodic
        boundary in `axis`, and `False` otherwise.

    """
    g = domain.buffered_graph(axis)

    # test for percolation from any node along axis
    percolate = False
    for btm in g:
        if btm[axis] == 0:
            top = list(btm)
            top[axis] = domain.mesh.shape[axis]
            top = tuple(top)

            try:
                if networkx.has_path(g,btm,top):
                    percolate = True
                    break
            except networkx.exception.NodeNotFound:
                # it is OK to raise this exception.
                # it just means the target node (top) is not in the domain.
                pass

    return percolate
