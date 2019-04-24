""" Perform simple simulations on meshes.
"""
from __future__ import division
import numpy as np
import fieldkit._simulate

def random_walk(domain, N, runs, steps, coords=None, images=None):
    """ Performs a simple random walk in a domain on a lattice.

    The random walk is executed on the nodes inside the
    :py:class:`~fieldkit.mesh.Domain`. One of the 6 lattice
    directions is chosen at random, and the walker moves to the
    new lattice site if it is also in the `domain`. Otherwise,
    it stays in place. The walk is performed in a sequence of
    `runs`, with each run having length `steps`. The walker
    coordinates are recorded before every run in an unwrapped
    system suitable for computing the mean-squared displacement.

    Parameters
    ----------
    domain : :py:class:`~fieldkit.mesh.Domain`
        The digitized domain to simulate in.
    N : int
        The number of random walkers.
    runs : int
        The number of runs to complete.
    steps : int
        The number of steps taken by the walker per run.
    coords : array_like or None
        An `Nx3` array of node (integer) coordinates.
    images : array_like or None
        An `Nx3` array of image (integer) coordinates.

    Returns
    -------
    trajectory : numpy.ndarray
        A `runsxNx3` array containing the unwrapped node coordinates.
    coords : numpy.ndarray
        An `Nx3` array containing the last wrapped coordinates.
    images : numpy.ndarray
        An `Nx3` array containing the last image coordinates.

    Notes
    -----
    The random walk uses the node coordinate system based on integers
    lying from `(0,0,0)` (inclusive) to `domain.mesh.shape` (exclusive).
    These coordinates can be transformed to real coordinates using the
    appropriate fractional conversion.

    The `coords` and `images` arguments can be used to restart a previous
    calculation. If used, `coords` must be properly wrapped to lie within
    the mesh-shape bounds. The `images` argument is optional; if not supplied,
    it will be assumed to be zero. However, this will not allow a clean
    restart for calculating things like the mean-squared displacement.

    The wrapped `coords` can be unwrapped by adding the appropriate images::

        unwrapped = coords + images * mesh.shape

    """
    # initial coordinates
    if coords is None:
        coords = np.asarray([domain.nodes[c] for c in np.random.randint(low=0, high=len(domain.nodes), size=N)], dtype=np.int32)
        coords = coords.transpose()
    else:
        coords = np.array(coords, dtype=np.int32)
        if coords.shape[0] != N or coords.shape[1] != 3:
            raise IndexError('Coordinate array must be Nx3')
        if not np.all([domain.mask[tuple(c)] for c in coords]):
            raise IndexError('All coordinates must lie in the domain')
        coords = coords.transpose()

    # initial images
    if images is None:
        images = np.zeros_like(coords)
    else:
        images = np.array(images, dtype=np.int32)
        if images.shape[0] != N or images.shape[1] != 3:
            raise IndexError('Image array must be Nx3')
        images = images.transpose()

    # output trajectory
    trajectory = np.zeros((runs,N,3))

    # run random walk simulation
    fieldkit._simulate.init_random_seed()
    for t in range(runs):
        trajectory[t] = coords.transpose() + images.transpose() * domain.mesh.shape
        fieldkit._simulate.random_walk(domain.mask, coords, images, steps)

    return trajectory,coords.transpose(),images.transpose()

def msd(trajectory, window, every=1):
    r""" Compute the mean-square displacement of a simulated trajectory.

    The mean-square displacement (MSD)

    .. math:: \langle (\mathbf{r}(t) - \mathbf{r}(0))^2 \rangle

    is evaluated using multiple time origins up to a maximum time (`window`).
    A time origin is taken *every* runs from the *trajectory*.
    The MSD at :math:`t=0` is trivially 0; the remainder of the MSD can
    be used to determine, e.g., the diffusion coefficient. This method
    resolves each component of the MSD independently.

    Parameters
    ----------
    trajectory : array_like
        A `runsxNx3` array-like object containing the unwrapped coordinates.
    window : int
        Time window (number of runs) for evaluating the MSD.
    every : int
        Number of runs between time origins. Default is 1.

    Returns
    -------
    rsq : numpy.ndarray
        A `(window+1)x3` NumPy array of the MSD.

    Examples
    --------
    The three-dimensional MSD can be determined by summing its components::

        >>> msd = fieldkit.simulate.msd(trajectory,window=10)
        >>> msd.shape
        (11,3)
        >>> rsq = np.sum(msd,axis=1)

    Notes
    -----
    This method internally calls a Fortran function to efficient perform the
    calculations. It will create a copy of the `trajectory` that is in a
    Fortran-efficient memory layout and shape using the double-precision
    floating point data type. Hence, the coordinates for `trajectory` can be
    any real-space positions, and do not need to be only the node coordinates
    returned by :py:meth:`random_walk`.

    """
    t = np.asfortranarray(np.rollaxis(trajectory,2),dtype=np.float64)
    rsq = fieldkit._simulate.msd(t,window,every)
    return rsq.transpose()

def msd_binned(trajectory, window, axis, bins, range, every=1):
    r""" Compute the spatially binned mean-square displacement of a simulated trajectory.

    The mean-square displacement (MSD)

    .. math:: \langle (\mathbf{r}(t) - \mathbf{r}(0))^2 | r_\alpha \rangle

    is evaluated using multiple time origins up to a maximum time (`window`).
    A time origin is taken *every* runs from the *trajectory*.
    The MSD at :math:`t=0` is trivially 0; the remainder of the MSD can
    be used to determine, e.g., the diffusion coefficient. This method
    resolves each component of the MSD independently.

    Unlike :py:math:`msd`, this method resolves the MSD based on the starting
    position of the particles in the trajectory, which is one method to compute the
    spatially varying diffusion coefficient. Particles are binned into a slab
    centered at :math:`r_\alpha`, where the axis :math:`\alpha` can be any of
    the Cartesian coordinates. The slabs are defined by a *range* along :math:`\alpha`
    and the number of *bins* in that range, similarly to the NumPy histogram method.
    The :math:`\alpha` coordinate of all particles in the *trajectory* must lie
    within *range*, or an error will be raised.

    Parameters
    ----------
    trajectory : array_like
        A `runsxNx3` array-like object containing the unwrapped coordinates.
    window : int
        Time window (number of runs) for evaluating the MSD.
    axis : int
        Cartesian coordinate (0, 1, or 2) for binning.
    bins : int
        The number of bins to use in the *range*.
    range : (float,float)
        A tuple defining the lower and upper bounds of all the bins.
    every : int
        Number of runs between time origins. Default is 1.

    Returns
    -------
    rsq : numpy.ndarray
        A `bins x (window+1) x 3` NumPy array of the MSD.
    edges : numpy.ndarray
        A `bins+1` NumPy array containing the bin edges, similar to NumPy histogram.

    Examples
    --------
    Compute the MSD using 4 bins of unit length along *z* ::

        >>> msd,edges = fieldkit.simulate.msd_binned(trajectory,window=10,axis=2,bins=4,range=(0,4))
        >>> msd.shape
        (5,11,3)
        >>> edges
        [0., 1.0, 2.0, 3.0, 4.0]

    The bin edges can be converted to bin centers ::

        >>> bins = 0.5*(edges[:-1]+edges[1:])
        [0.5, 1.5, 2.5, 3.5]

    Notes
    -----
    This method internally calls a Fortran function to efficient perform the
    calculations. It will create a copy of the `trajectory` that is in a
    Fortran-efficient memory layout and shape using the double-precision
    floating point data type. Hence, the coordinates for `trajectory` can be
    any real-space positions, and do not need to be only the node coordinates
    returned by :py:meth:`random_walk`.

    """
    # ensure all coordinates lie within range
    if np.any(trajectory[...,axis] < range[0]) or np.any(trajectory[...,axis] >= range[1]):
        raise ValueError('All coordinates along axis {} must lie in range [{},{})'.format(axis,range[0],range[1]))

    t = np.asfortranarray(np.rollaxis(trajectory,2),dtype=np.float64)
    rsq = fieldkit._simulate.msd_binned(t,axis,bins,range[0],range[1],window,every)
    edges = np.linspace(range[0],range[1],bins+1)

    return rsq.transpose(),edges
