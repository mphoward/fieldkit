""" Data structures for manipulating membrane data.

"""
from __future__ import division
import numpy as np

__all__ = ["Mesh","Field"]

class Mesh(object):
    """ Mesh

    The mesh geometry is a three-dimensional triclinic periodic cell.

    Attributes
    ----------
    grid
    lattice
    shape
    dim
    step : array_like
        Step size of the mesh in each dimension.

    """
    def __init__(self):
        self._grid = None

    def from_lattice(self, N, L, tilt=None):
        """ Initialize mesh from a lattice.

        `N` lattice points are placed along each lattice vector.
        The edge lengths of the orthorhombic box are given by `L`,
        and the box is deformed by the `tilt` factors.

        Parameters
        ----------
        N : int or array_like
            Number of lattice points.
        L : float or array_like
            Edge lengths of undeformed orthorhombic box.
        tilt : None or array_like
            If specified, the tilt factors for the lattice.

        Returns
        -------
        :py:obj:`Mesh`
            A reference to the mesh object.

        """
        N = np.asarray(N, dtype=np.int32)
        try:
            if len(N) != 3:
                raise IndexError('Meshes must be 3D')
        except TypeError:
            N = np.full(3, N, dtype=np.int32)

        L = np.asarray(L)
        try:
            if len(L) != len(N):
               raise IndexError('Step size must match grid size')
        except TypeError:
            L = np.full(len(N),L)

        if tilt is not None:
            try:
                if len(tilt) == 3:
                    tilt = np.array(tilt)
                else:
                    raise TypeError('Tilt factors must be 3D array')
            except:
                raise TypeError('Tilt factors must be 3D array')
        else:
            tilt = np.zeros(3)

        # fill grid points using the fractional coordinates and hoomd/lammps triclinic cell
        self._lattice = np.diag(L)
        self._lattice[0,1] = tilt[0] * L[1]
        self._lattice[0,2] = tilt[1] * L[2]
        self._lattice[1,2] = tilt[2] * L[2]
        self._grid = np.empty(np.append(N,len(N)))
        for n in np.ndindex(self._grid.shape[:-1]):
            self._grid[n] = np.dot(self._lattice, n/N)

        # step spacing along each cartesian axis
        self.step = np.zeros(self.dim)
        for i in range(self.dim):
            origin = [0] * self.dim
            index = list(origin)
            index[i] = 1
            dr = self._grid[tuple(index)] - self._grid[tuple(origin)]
            self.step[i] = np.sqrt(np.sum(dr*dr))

        return self

    def from_array(self, grid):
        """ Initialize mesh from an array of data.

        Use an existing grid to define the mesh. The grid should be
        a four-dimensional array. The first three dimensions should have
        sizes corresponding to the number of points in *x*, *y*, and *z*.
        The last dimension should be a 3-element tuple giving the grid
        coordinates in real space. The *x* index is thus the slowest varying
        coordinate, while the *z* index is the fastest varying one.

        Parameters
        ----------
        grid : array_like
            Four-dimensional array to initialize the grid from.

        Returns
        -------
        :py:obj:`Mesh`
            A reference to the mesh object.

        """
        grid = np.asarray(grid)
        self._grid = np.copy(grid)

        if self.dim != 3:
            raise IndexError('Only 3D grids are supported')

        # step spacing along each cartesian axis
        self.step = np.zeros(self.dim)
        for i in range(self.dim):
            origin = [0] * self.dim
            index = list(origin)
            index[i] = 1
            dr = self._grid[tuple(index)] - self._grid[tuple(origin)]
            self.step[i] = np.sqrt(np.sum(dr*dr))

        # convert box extent into lattice vectors
        a = self._grid[-1,0,0] - self._grid[0,0,0]
        b = self._grid[0,-1,0] - self._grid[0,0,0]
        c = self._grid[0,0,-1] - self._grid[0,0,0]
        # extend the lattice vectors to next unit cell by one step
        a += self.step[0] * (a / np.linalg.norm(a))
        b += self.step[1] * (b / np.linalg.norm(b))
        c += self.step[2] * (c / np.linalg.norm(c))
        self._lattice = np.column_stack((a,b,c))

        return self

    def from_file(self, filename):
        """ Initialize mesh from a saved NumPy file.

        This method is a convenience wrapper for :py:meth:`from_array`.

        Parameters
        ----------
        filename : str
            NumPy file containing coordinates.

        Returns
        -------
        :py:obj:`Mesh`
            A reference to the mesh object.

        """
        grid = np.load(filename)
        return self.from_array(grid)

    @property
    def lattice(self):
        r""" Cell matrix corresponding to the periodic cell

        Gives the matrix `h` that transforms a fractional lattice coordinate
        to a real-space coordinate in the periodic cell. Each column of the
        matrix is a real-space lattice vector.

        Returns
        -------
        h : array_like
            Transformation matrix

        Dotting a fractional coordinate into this matrix yields the real space
        coordinate.

        """
        return self._lattice

    @property
    def grid(self):
        """ Coordinates of the grid points in the mesh.

        Returns
        -------
        array_like:
            A four-dimensional grid containing the mesh points.

        """
        return self._grid

    def __getitem__(self, index):
        return self._grid[index]

    @property
    def shape(self):
        """ Shape of the mesh.

        Returns
        -------
        array_like:
            A tuple containing the size of the mesh in each dimension.

        """
        return self._grid.shape[:-1]

    @property
    def dim(self):
        """ Dimensionality of the mesh.

        Returns
        -------
        int:
            Number of dimensions spanned by the mesh.

        """
        return len(self.shape)

    @property
    def L(self):
        """ Length of the periodic simulation cell along each lattice vector.

        Returns
        -------
        array_like:
            Length of the simulation cell.

        """
        return np.linalg.norm(self.lattice,axis=0)

    def neighbors(self, n, full=True):
        """ Get the indexes of neighboring nodes subject to periodic boundaries.

        Parameters
        ----------
        n : array_like
            Tuple giving the index in the mesh to find neighbors for.
        full : bool
            If True, return all 6 adjacent neighbors. Otherwise, only
            return the 3 in the "forward" directions on the lattice.

        Returns
        -------
        tuple:
            A tuple of tuple indexes in the mesh corresponding to the
            neighbors of `n`.

        """
        i,j,k = n
        neighs = [((i+1) % self.shape[0], j, k),
                  (i, (j+1) % self.shape[1], k),
                  (i, j, (k+1) % self.shape[2])]
        if full:
            neighs += [((i-1) % self.shape[0], j, k),
                       (i, (j-1) % self.shape[1], k),
                       (i, j, (k-1) % self.shape[2])]
        return tuple(neighs)

class Field(object):
    """ Scalar field on a :py:obj:`~Mesh`.

    Parameters
    ----------
    mesh : :py:obj:`~Mesh`
        Mesh used to define the volume for the field.

    Attributes
    ----------
    field
    shape

    Examples
    --------
    Values of the field can be accessed directly by index::

        field[0,:,-1]

    """
    def __init__(self, mesh):
        self._mesh = mesh
        self._field = np.zeros(self._mesh.shape)

    def from_array(self, field, index=None, axis=None):
        """ Initialize field data from an array.

        The `field` data can be a three or four dimensional array.
        It is copied directly if it is three dimensional, and must
        match the shape of the `mesh`. If it is four-dimensional,
        `index` and `axis` can be applied to slice the appropriate
        data using `np.take()`.

        Parameters
        ----------
        field : array_like
            Array of field data.
        index : None or int
            If specified, take from `field` at `index`.
        axis : None or int
            If specified, use `axis` when selecting `index` to take.

        Returns
        -------
        :py:obj:`~Field`
            A reference to the field object.

        """
        field = np.asarray(field)
        if index is not None:
            field = field.take(indices=index, axis=axis)

        self.field = field
        return self

    def from_file(self, filename, index=None, axis=None):
        """ Initialize field data from a file.

        The `field` data can be a three or four dimensional array.
        It is copied directly if it is three dimensional, and must
        match the shape of the `mesh`. If it is four-dimensional,
        `index` and `axis` can be applied to slice the appropriate
        data using `np.take()`. This method is a convenience wrapper
        around :py:meth:`~from_array`.

        Parameters
        ----------
        filename : str
            NumPy file containing the field data.
        index : None or int
            If specified, take from `field` at `index`.
        axis : None or int
            If specified, use `axis` when selecting `index` to take.

        Returns
        -------
        :py:obj:`~Field`
            A reference to the field object.

        """
        field = np.load(filename)
        return self.from_array(field, index, axis)

    @property
    def field(self):
        """ Values of the field on the input mesh.
        """
        return self._field

    @field.setter
    def field(self, field):
        """ Sets the field from an existing array.

        The shape of the field must be consistent with the
        mesh the field was initialzed with.

        Parameters
        ----------
        field : array_like
            Three-dimensional field values to set

        Raises
        ------
        TypeError
            If the field shape does not match the mesh shape.

        """
        field = np.asarray(field)
        if field.shape == self._mesh.shape:
            self._field = np.copy(field)
        else:
            raise TypeError('Field shape is not appropriate for mesh')

    @property
    def shape(self):
        """ Shape of the field.

        The field shape matches the underlying mesh shape.

        Returns
        -------
        array_like
            Tuple giving the number of points along each mesh dimension.

        """
        return self._field.shape

    def __getitem__(self, index):
        return self._field[index]
