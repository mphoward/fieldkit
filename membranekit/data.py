""" Data structures for manipulating membrane data.

"""
__all__ = ["Mesh"]

import numpy as np

class Mesh(object):
    """ Mesh

    The mesh geometry is a three-dimensional triclinic periodic cell.

    Attributes
    ----------
    cell_matrix
    grid
    shape
    dim
    length
    tilt
    step : array_like
        Step size of the mesh in each dimension.

    """
    def __init__(self):
        self._L = None
        self._tilt = None
        self._grid = None

    def from_lattice(self, N, length, tilt=None):
        """ Initialize mesh from a lattice.

        `N` lattice points are placed along each lattice vector.
        The nearest distances between faces are given by `length`
        and the box is deformed by the `tilt` factors.

        Parameters
        ----------
        N : int or array_like
            Number of lattice points.
        length : float or array_like
            Nearest distances between edges of the lattice.
        tilt : None or array_like
            If specified, the tilt factors for the lattice.

        Returns
        -------
        :py:obj:`Mesh`
            A reference to the mesh object.

        """
        N = np.asarray(N, dtype=np.int32)
        try:
            len(N) == 3
        except TypeError:
            N = np.full(3, N, dtype=np.int32)

        L = np.asarray(length)
        try:
            if len(L) == len(N):
                self._L = L
            else:
               raise IndexError('Step size must match grid size')
        except TypeError:
            self._L = np.full(len(N),L)

        if tilt is not None:
            try:
                if len(tilt) == 3:
                    self._tilt = np.array(tilt)
                else:
                    raise TypeError('Tilt factors must be 3D array')
            except:
                raise TypeError('Tilt factors must be 3D array')
        else:
            self._tilt = np.zeros(3)

        # fill grid points using the fractional coordinates
        h = self.cell_matrix
        self._grid = np.empty(np.append(N,len(N)))
        for n in np.ndindex(self._grid.shape[:-1]):
            self._grid[n] = np.dot(h, n/N)

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

        # convert box extent into length and tilt factors
        a = self._grid[-1,0,0] - self._grid[0,0,0]
        b = self._grid[0,-1,0] - self._grid[0,0,0]
        c = self._grid[0,0,-1] - self._grid[0,0,0]
        # extend the lattice vectors to next unit cell by one step
        a += self.step[0] * (a / np.linalg.norm(a))
        b += self.step[1] * (b / np.linalg.norm(b))
        c += self.step[2] * (c / np.linalg.norm(c))

        self._L = np.array([a[0], b[1], c[2]])
        self._tilt = np.array([b[0]/self._L[1], c[0]/self._L[2], c[1]/self._L[2]])

        return self

    @property
    def cell_matrix(self):
        r""" Cell matrix corresponding to the periodic cell

        Gives the matrix `h` that transforms a fractional lattice coordinate
        to a real-space coordinate in the periodic cell.

        Returns
        -------
        h : array_like
            Transformation matrix

        Notes
        -----

        The mesh :py:attr:`~length` and :py:attr:`~tilt` define a transformation
        matrix for the periodic simulation cell.

        .. math::

            \begin{pmatrix}
            L_x & t_{xy} L_y & t_{xz} L_z \\
            0   &        L_y & t_{yz} L_z \\
            0   &            &        L_z
            \end{pmatrix}

        where **L** is the vector of lengths along each lattice vector and **t**
        is the vector of tilt factors.

        Dotting a fractional coordinate into this matrix yields the real space
        coordinate.

        """
        L = self.length
        h = np.diag(L)
        h[0,1] = self.tilt[0] * L[1]
        h[0,2] = self.tilt[1] * L[2]
        h[1,2] = self.tilt[2] * L[2]

        return h

    @property
    def grid(self):
        """ Coordinates of the grid points in the mesh.

        Returns
        -------
        array_like:
            A four-dimensional grid containing the mesh points.

        """
        return self._grid

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
    def length(self):
        """ Length of the periodic simulation cell.

        Returns
        -------
        array_like:
            Length of the simulation cell along each lattice vector.

        """
        return self._L

    @property
    def tilt(self):
        """ Fractional tilt factors.

        Returns
        -------
        array_like:
            The fractional tilt factors for a triclinic cell.

        For an orthorhombic simulation cell, all tilt factors are zero.

        """
        return self._tilt
