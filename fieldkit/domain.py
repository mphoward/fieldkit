""" Identify and characterize domains.
"""
from __future__ import division
import numpy as np
import networkx
import skimage.measure
from fieldkit.mesh import TriangulatedSurface

def find(field, threshold):
    """ Finds connected domains in a field.

    A domain is defined to be a connected region of lattice
    points having a `field` value greater than or equal to
    `threshold`, subject to periodic boundary conditions of the
    lattice.

    Parameters
    ----------
    field : :py:class:`~fieldkit.mesh.Field`
        The field to analyze for continuous domains.
    threshold : float
        Threshold tolerance for the field to consider a lattice
        site "filled".

    Returns
    -------
    iterator
        An iterator for indexes in :py:class:`~fieldkit.mesh.Mesh`.
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

    Todo
    ----
    This method will eventually be extended to subsets of the nodes in a domain.

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

    Todo
    -----
    This method will eventually be extended to subsets of the nodes in a domain. It also needs to be validated more
    thoroughly in periodic boundary conditions.

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

def burn(field, threshold):
    """ Compute the burn number distribution and medial axis for a domain.

    Implements the burning algorithm + medial axis protocol outlined by
    Lindquist et al. [1]_ A burning algorithm ignites a "flame" from the
    boundaries of the domain, and marches the flame front through. The medial
    axis is defined by the locus of voxels where the flame front is extinguished
    by an opposing front.

    Parameters
    ----------
    field : :py:class:`~fieldkit.mesh.Field`
        The field to burn.
    threshold : float
        Threshold tolerance for the field to consider a lattice site "filled".

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
    burn_number = np.zeros(field.shape, dtype=np.int32)
    burn_number[field.field >= threshold] = BURN_SIGNAL

    medial_axis = np.zeros(field.shape, dtype=bool)

    # iterate through while some burn numbers have not been determined
    k = 1
    while np.any(burn_number == BURN_SIGNAL):
        burn_vecs = {}
        for pt in np.ndindex(burn_number.shape):
            # only work on unchecked points
            if burn_number[pt] != BURN_SIGNAL:
                continue

            # get neighboring points and burn vectors
            neighs = field.mesh.neighbors(pt, full=True)
            burn_vecs[pt] = []
            for n in neighs:
                if burn_number[n] == k-1:
                    # this is minimum image on the mesh
                    dn = np.array(pt) - np.array(n)
                    dn -= field.shape * np.round(dn.astype(float)/field.shape).astype(int)
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
                    n = tuple((pt + v) % field.shape)
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
                            v1 = tuple((pt + vi - vj) % field.shape)
                            v2 = tuple((pt - vi + vj) % field.shape)
                            if burn_number[v1] == k-1 and burn_number[v2] == k-1:
                                medial_axis[pt] = True
        k += 1


    # extract the indices corresponding to the medial axis
    axis = np.moveaxis(np.indices(field.shape), 0, -1)[medial_axis]

    return burn_number, axis
