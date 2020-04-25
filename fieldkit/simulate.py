""" Perform simple simulations on meshes.
"""
from __future__ import division
import os
import random
import numpy as np
import fieldkit._simulate

def random_walk(domain, N, runs, steps, coords=None, images=None, seed=None, threads=None):
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
    seed : int
        The seed to use for the random number generator. If None,
        a value is drawn from the system entropy.
    coords : array_like or None
        An `Nx3` array of node (integer) coordinates.
    images : array_like or None
        An `Nx3` array of image (integer) coordinates.
    threads : int
        The number of OpenMP threads to use in the simulation. If None,
        use the value of `OMP_NUM_THREADS` from the environment if it is
        set; otherwise, default to 1.

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
    If restarting a run, it is advisable to choose a new `seed` as the state
    of the random number generator is not preserved between calls.

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

    # if unspecified, draw a seed from the system entropy
    if seed is None:
        seed = random.SystemRandom().randint(-2147483648,2147483647)

    # try to get threads
    if threads is None:
        try:
            threads = os.environ['OMP_NUM_THREADS']
        except:
            threads = 1

    # run random walk simulation
    trajectory = fieldkit._simulate.random_walk(domain.mask, coords, images, steps, runs, seed, threads)

    return trajectory.transpose(),coords.transpose(),images.transpose()

def biased_walk(domain, probs, N, runs, steps, coords=None, images=None, seed=None, threads=None):
    r""" Performs a biased random walk in a domain on a lattice.

    The biased random walk is executed on the nodes inside the
    :py:class:`~fieldkit.mesh.Domain`. One of the 6 lattice
    directions is chosen accordings to the weights in `probs`,
    and the walker moves to the new lattice site if it is also
    in the `domain`. Otherwise, it stays in place. The walk is
    performed in a sequence of `runs`, with each run having
    length `steps`. The walker coordinates are recorded before
    every run in an unwrapped system suitable for computing the
    mean-squared displacement.

    Parameters
    ----------
    domain : :py:class:`~fieldkit.mesh.Domain`
        The digitized domain to simulate in.
    probs : array_like
        The transition probabilities to 6 adjacent lattice sites,
        as an `(Nx,Ny,Nz,6)` array.
    N : int
        The number of random walkers.
    runs : int
        The number of runs to complete.
    steps : int
        The number of steps taken by the walker per run.
    seed : int
        The seed to use for the random number generator. If None,
        a value is drawn from the system entropy.
    coords : array_like or None
        An `Nx3` array of node (integer) coordinates.
    images : array_like or None
        An `Nx3` array of image (integer) coordinates.
    threads : int or None
        The number of OpenMP threads to use in the simulation. If None,
        use the value of `OMP_NUM_THREADS` from the environment if it is
        set; otherwise, default to 1.

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
    This method is very similar to :py:func:`random_walk`, other than
    the transition probabilites, so refer to its documentation for the
    meaning of the coordinates and images.

    The transition probabilites (`probs`) give the weighted probabilities
    to transition to an adjacent lattice site. They should be specified in
    the order `(+x,-x,+y,-y,+z,-z)` for each site. Each entry must be in the
    range [0,1], and the sum of the transition probabilities for a site must
    be less than or equal to 1. If the sum :math:`p` is less than 1 for a site,
    the walker will remain on the site with probability :math:`1-p`.

    This is effectively an implementation of null-event (rejection) kinetic
    Monte Carlo. To mapping from kMC transition rates :math:`\Gamma_{ij}` to
    transition probabilities :math:`p_{ij}`, multiply the transition rates by
    a nominal average timestep :math:`\Delta t` (:math:`p_{ij} = \Delta t \Gamma_{ij}`).
    This timestep should be chosen in a way that makes the algorithm efficient
    and still satisfies the conditions for the transition probabilities. One way
    to do this is to find the **maximum sum** of transition rates for all
    sites :math:`\Gamma_i^*`. The timestep is then :math:`\Delta t = 1/\Gamma^*`.

    To create an unbiased random walk::

        probs = np.full(list(mesh.shape) + [6], 1./6.)

    The timestep is then :math:`\Delta t = \Delta x^2/6 D_0` where `D_0` is the
    diffusion coefficient on the unconfined lattice.

    """
    # preprocess the transition probabilities
    if np.any(probs) < 0 or np.any(probs) > 1:
        raise ValueError('Transition probabilities must lie between 0 and 1.')
    cumprobs = np.cumsum(probs, axis=-1, dtype=np.float64)
    if np.any(cumprobs[...,-1]) > 1:
        raise ValueError('Transition probabilities must sum <= 1.')
    cumprobs = np.asfortranarray(np.moveaxis(cumprobs,-1,0))

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

    # if unspecified, draw a seed from the system entropy
    if seed is None:
        seed = random.SystemRandom().randint(-2147483648,2147483647)

    # try to get threads
    if threads is None:
        try:
            threads = os.environ['OMP_NUM_THREADS']
        except:
            threads = 1

    # run random walk simulation
    trajectory = fieldkit._simulate.biased_walk(domain.mask, cumprobs, coords, images, steps, runs, seed, threads)

    return trajectory.transpose(),coords.transpose(),images.transpose()

def kmc(domain, rates, samples, N, steps, coords=None, images=None, times=None, seed=None, threads=None):
    r""" Performs a biased random walk on a lattice using kinetic Monte Carlo.

    The biased random walk is executed on the nodes inside the
    :py:class:`~fieldkit.mesh.Domain`. One of the 6 lattice
    directions is chosen and unconditionally accepted if the
    move keeps the walker in the `domain`, and the time counter
    is advanced by a random amount based on the exponential
    distribution for a Markov process. The walk is performed in a
    sequence of `steps`, and the walker coordinates are recorded
    at every time point in `samples` in an unwrapped system
    suitable for computing the mean-squared displacement.

    Parameters
    ----------
    domain : :py:class:`~fieldkit.mesh.Domain`
        The digitized domain to simulate in.
    rates : array_like
        The transition rates to 6 adjacent lattice sites,
        as an `(Nx,Ny,Nz,6)` array.
    samples : array_like
        Time points to sample trajectory at.
    N : int
        The number of random walkers.
    steps : int
        Maximum number of steps taken by the walker per run.
    seed : int
        The seed to use for the random number generator. If None,
        a value is drawn from the system entropy.
    coords : array_like or None
        An `Nx3` array of node (integer) coordinates.
    images : array_like or None
        An `Nx3` array of image (integer) coordinates.
    times : array_like or None
        An `N` array of time counters.
    threads : int or None
        The number of OpenMP threads to use in the simulation. If None,
        use the value of `OMP_NUM_THREADS` from the environment if it is
        set; otherwise, default to 1.

    Returns
    -------
    trajectory : numpy.ndarray
        A `runsxNx3` array containing the unwrapped node coordinates and current time counters.
    coords : numpy.ndarray
        An `Nx3` array containing the last wrapped coordinates.
    images : numpy.ndarray
        An `Nx3` array containing the last image coordinates.
    times : numpy.ndarray
        An `N` array containing the last time counters.

    Notes
    -----
    This method is very similar to :py:func:`random_walk`, other than
    the transition rates, so refer to its documentation for the
    meaning of the coordinates and images.

    The transition rates (`rates`) give the weighted probabilities (per time)
    to transition to an adjacent lattice site. They should be specified in
    the order `(+x,-x,+y,-y,+z,-z)` for each site. The rates effectively
    define the unit of time. The algorithm implements rejection-free kinetic
    Monte Carlo, so each walker gets its own time counter that advances it
    by a stochastic amount based on the local transition rates. This can be
    more efficient than :py:func:`biased_walk` when there are significant
    disparities in transition rates between different parts of the lattice
    (i.e., some rare events). However, it comes at the cost of drawing an
    additional random number. Usually, :py:func`kmc` is faster than
    :py:func:`biased_walk` if at least 50% of the moves would be rejected
    by that method.

    Because the time counter advances randomly, the walker coordinates are
    interpolated between steps to save them at the time points specified
    by `samples`. A maximum number of `steps` are attempted; if this number
    is insufficient for sampling, a `RuntimeError` will be raised so that
    the simulation can be restarted after catching the exception.

    """
    # preprocess the transition probabilities
    if np.any(rates) < 0:
        raise ValueError('Transition probabilities must be non-negative.')
    cumrates = np.cumsum(rates, axis=-1, dtype=np.float64)
    cumrates = np.asfortranarray(np.moveaxis(cumrates,-1,0))

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

    # initial times
    if times is None:
        times = np.zeros(N, dtype=np.float64)
    else:
        times = np.array(times, dtype=np.float64)

    # if unspecified, draw a seed from the system entropy
    if seed is None:
        seed = random.SystemRandom().randint(-2147483648,2147483647)

    # try to get threads
    if threads is None:
        try:
            threads = os.environ['OMP_NUM_THREADS']
        except:
            threads = 1

    # make sorted array of time points
    samples = np.array(samples, dtype=np.float64)
    samples.sort()
    if np.any(times > samples[0]):
        raise RuntimeError('Current walker times must be less than first sample time.')

    # run random walk simulation
    trajectory = fieldkit._simulate.kmc(domain.mask, cumrates, coords, images, times, samples, steps, seed, threads)

    if np.any(times < samples[-1]):
        raise RuntimeError('Number of steps not sufficient to sample.')

    return trajectory.transpose(),coords.transpose(),images.transpose(),times

def msd(trajectory, window, every=1, threads=None):
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
    threads : int or None
        The number of OpenMP threads to use in the calculation. If None,
        use the value of `OMP_NUM_THREADS` from the environment if it is
        set; otherwise, default to 1.

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
    # try to get threads
    if threads is None:
        try:
            threads = os.environ['OMP_NUM_THREADS']
        except:
            threads = 1

    t = np.asfortranarray(np.rollaxis(trajectory,2),dtype=np.float64)
    rsq = fieldkit._simulate.msd(t,window,every,threads)
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

    Unlike :py:func:`msd`, this method resolves the MSD based on the starting
    position of the particles in the trajectory, which is one method to compute the
    spatially varying diffusion coefficient. Particles are binned into a slab
    centered at :math:`r_\alpha`, where the axis :math:`\alpha` can be any of
    the Cartesian coordinates. The slabs are defined by a *range* along :math:`\alpha`
    and the number of *bins* in that range, similarly to the NumPy histogram method.
    The calculation is only performed on time origins in the *trajectory* when the
    :math:`\alpha` coordinate of a particles lies within *range*.

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
    t = np.asfortranarray(np.rollaxis(trajectory,2),dtype=np.float64)
    rsq = fieldkit._simulate.msd_binned(t,axis,bins,range[0],range[1],window,every)
    edges = np.linspace(range[0],range[1],bins+1)

    return rsq.transpose(),edges

def msd_survival(trajectory, window, axis, bins, range, every=1):
    r""" Compute the spatially binned mean-square displacement of a simulated trajectory.

    The condition mean-square displacement (MSD)

    .. math:: \langle (\mathbf{r}(t) - \mathbf{r}(0))^2 | r_\alpha \rangle

    is evaluated using multiple time origins up to a maximum time (`window`).
    A time origin is taken *every* runs from the *trajectory*.
    The MSD at :math:`t=0` is trivially 0; the remainder of the MSD can
    be used to determine, e.g., the diffusion coefficient. This method
    resolves each component of the MSD that is normal to *axis* independently.

    Unlike :py:func:`msd`, this method resolves the MSD based on the position of
    of the particles throughout the trajectory. Particles are binned into a slab
    centered at :math:`r_\alpha`, where the axis :math:`\alpha` can be any of
    the Cartesian coordinates. The slabs are defined by a *range* along :math:`\alpha`
    and the number of *bins* in that range, similarly to the NumPy histogram method.
    The calculation is only performed on time origins in the *trajectory* when the
    :math:`\alpha` coordinate of a particles lies within *range*.

    Unlike :py:func:`msd_binned`, this method does not compute the "true" MSD, but
    one that is conditionally averaged. The relationship between the computed MSD
    and the diffusion coefficient is:

    .. math::

        \langle \Delta r_\beta^2 | r_\alpha \rangle = 2 D_{\beta\beta} t P(t)

    where :math:`P(t)` is the survival probability. Given the number of counts
    in a bin at a given time :math:`N(t)`, the probability is

    .. math:: P(t) = \frac{N(t)}{N{0}}

    Both the MSD and *N* are returned by the method, allowing for *D* to be
    extracted.

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
        A `bins x (window+1) x 2` NumPy array of the MSD normal to *axis*.
    counts : numpy.ndarray
        A `bins x (window+1)` NumPy
    edges : numpy.ndarray
        A `bins+1` NumPy array containing the bin edges, similar to NumPy histogram.

    Examples
    --------
    Compute the MSD using 4 bins of unit length along *z* ::

        >>> msd,counts,edges = fieldkit.simulate.msd_survival(trajectory,window=10,axis=2,bins=4,range=(0,4))
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
    t = np.asfortranarray(np.rollaxis(trajectory,2),dtype=np.float64)
    rsq,counts = fieldkit._simulate.msd_survival(t,axis,bins,range[0],range[1],window,every)
    rsq = np.delete(rsq, axis, 0)
    edges = np.linspace(range[0],range[1],bins+1)

    return rsq.transpose(),counts.transpose(),edges

def msd_survival_cylinder(radial, axial, window, bins, range, every=1):
    r""" Compute the radially binned axial mean-square displacement.

    The condition mean-square displacement (MSD)

    .. math:: \langle (\mathbf{z}(t) - \mathbf{z}(0))^2 | r \rangle

    is evaluated using multiple time origins up to a maximum time (`window`).
    A time origin is taken *every* runs from the *trajectory*.
    The MSD at :math:`t=0` is trivially 0; the remainder of the MSD can
    be used to determine, e.g., the diffusion coefficient. This method
    resolves only the axial component of the MSD.

    Unlike :py:func:`msd_survival`, this method resolves the MSD based on the
    radial position of of the particles in a cylindrical geometry that the caller
    defines. Particles are binned into a slab centered at :math:`r`, and the slabs
    are defined by a *range* and the number of *bins* in that range, similarly to
    the NumPy histogram method. The calculation is only performed on time origins
    in *axial* when the *radial* coordinate of a particles lies within *range*.

    This method does not compute the "true" MSD, but one that is conditionally averaged.
    The relationship between the computed MSD and the diffusion coefficient is:

    .. math::

        \langle \Delta z^2 | r \rangle = 2 D_{zz} t P(t)

    where :math:`P(t)` is the survival probability. Given the number of counts
    in a bin at a given time :math:`N(t)`, the probability is

    .. math:: P(t) = \frac{N(t)}{N{0}}

    Both the MSD and *N* are returned by the method, allowing for *D* to be
    extracted.

    Parameters
    ----------
    radial : array_like
        A `runsxN` array-like object containing the radial coordinates.
    axial : array_like
        A `runsxN` array-like object containing the axial coordinates.
    window : int
        Time window (number of runs) for evaluating the MSD.
    bins : int
        The number of bins to use in the *range*.
    range : (float,float)
        A tuple defining the lower and upper bounds of all the bins.
    every : int
        Number of runs between time origins. Default is 1.

    Returns
    -------
    rsq : numpy.ndarray
        A `bins x (window+1)` NumPy array of the axial MSD.
    counts : numpy.ndarray
        A `bins x (window+1)` NumPy
    edges : numpy.ndarray
        A `bins+1` NumPy array containing the bin edges, similar to NumPy histogram.

    Examples
    --------
    Compute the MSD using 4 bins of unit length along *r* ::

        >>> msd,counts,edges = fieldkit.simulate.msd_survival(r,z,window=10,bins=4,range=(0,4))
        >>> msd.shape
        (5,11)
        >>> edges
        [0., 1.0, 2.0, 3.0, 4.0]

    The bin edges can be converted to bin centers ::

        >>> bins = 0.5*(edges[:-1]+edges[1:])
        [0.5, 1.5, 2.5, 3.5]

    Notes
    -----
    This method internally calls a Fortran function to efficiently perform the
    calculations. It will create a copy of the `radial` and `axial` coordinates
    that is in a Fortran-efficient memory layout and shape using the double-precision
    floating point data type. Hence, the coordinates can be any real-space positions,
    and do not need to be only the node coordinates returned by :py:meth:`random_walk`.

    """
    r = np.asfortranarray(radial,dtype=np.float64)
    z = np.asfortranarray(axial,dtype=np.float64)
    rsq,counts = fieldkit._simulate.msd_survival_cylinder(r,z,bins,range[0],range[1],window,every)
    edges = np.linspace(range[0],range[1],bins+1)

    return rsq.transpose(),counts.transpose(),edges
