""" Data structures for working with meshes.

"""
from __future__ import division
import numpy as np
import scipy.interpolate
from fieldkit.lattice import Lattice

__all__ = ["Mesh","Field","TriangulatedSurface"]

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
        self._lattice = None

    def from_lattice(self, N, lattice):
        """ Initialize mesh from a lattice.

        `N` lattice points are placed along each `lattice` vector.

        Parameters
        ----------
        N : int or array_like
            Number of lattice points.
        lattice : :py:class:`~fieldkit.lattice.Lattice`
            Lattice to initialize with.

        Returns
        -------
        :py:class:`Mesh`
            A reference to the mesh object.

        """
        N = np.asarray(N, dtype=np.int32)
        try:
            if len(N) != 3:
                raise IndexError('Meshes must be 3D')
        except TypeError:
            N = np.full(3, N, dtype=np.int32)

        # set lattice object and grid
        self._lattice = lattice
        self._grid = np.empty(np.append(N,len(N)))

        # fill grid points using the fractional coordinates and lattice
        self.step = self.lattice.L / N
        for n in np.ndindex(self.grid.shape[:-1]):
            self.grid[n] = self.lattice.as_coordinate(n/N)

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
        :py:class:`Mesh`
            A reference to the mesh object.

        """
        grid = np.asarray(grid)
        self._grid = np.copy(grid)

        if self.dim != 3:
            raise IndexError('Only 3D grids are supported')

        if np.any(np.array(self.shape) == 1):
            raise IndexError('At least 2 nodes are required per grid dimension')

        # step spacing along each cartesian axis
        self.step = np.zeros(self.dim)
        for i in range(self.dim):
            origin = [0] * self.dim
            index = list(origin)
            index[i] = 1
            dr = self.grid[tuple(index)] - self.grid[tuple(origin)]
            self.step[i] = np.sqrt(np.sum(dr*dr))

        # convert box extent into lattice vectors
        a = self.grid[-1,0,0] - self.grid[0,0,0]
        b = self.grid[0,-1,0] - self.grid[0,0,0]
        c = self.grid[0,0,-1] - self.grid[0,0,0]
        # extend the lattice vectors to next unit cell by one step
        a += self.step[0] * (a / np.linalg.norm(a))
        b += self.step[1] * (b / np.linalg.norm(b))
        c += self.step[2] * (c / np.linalg.norm(c))
        self._lattice = Lattice(a,b,c)

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
        :py:class:`Mesh`
            A reference to the mesh object.

        """
        grid = np.load(filename)
        return self.from_array(grid)

    @property
    def lattice(self):
        """ Lattice corresponding to the periodic cell

        Returns
        -------
        :py:class:`~fieldkit.lattice.Lattice`
            Lattice object representing the periodic cell.

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

        neighs = []
        if self.shape[0] > 1:
            neighs.append(((i+1) % self.shape[0], j, k))
        if full and self.shape[0] > 2:
            neighs.append(((i-1) % self.shape[0], j, k))
        if self.shape[1] > 1:
            neighs.append((i, (j+1) % self.shape[1], k))
        if full and self.shape[1] > 2:
            neighs.append((i, (j-1) % self.shape[1], k))
        if self.shape[2] > 1:
            neighs.append((i, j, (k+1) % self.shape[2]))
        if full and self.shape[2] > 2:
            neighs.append((i, j, (k-1) % self.shape[2]))
        return tuple(neighs)

class Field(object):
    """ Scalar field on a :py:class:`~Mesh`.

    Parameters
    ----------
    mesh : :py:class:`~Mesh`
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
        :py:class:`~Field`
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
        :py:class:`~Field`
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
    def mesh(self):
        """ Mesh corresponding to the field.

        Returns
        -------
        :py:class:`~Mesh`
            The mesh attached to the field.

        """
        return self._mesh

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

    def __setitem__(self, index, item):
        self._field[index] = item

    def buffered(self):
        """ Create a copy of the field data buffered with the periodic boundary nodes.

        The data is the same as that in :py:attr:`~field`, but it is extended by 1 node
        to include the first points from the next (positive) periodic cells, and fractional
        coordinates up to and including 1 are now available.

        Returns
        -------
        array_like
            The buffered field data.

        """
        # for interpolation, clone the first row into the last row
        field = np.empty((self.mesh.shape[0]+1,self.mesh.shape[1]+1,self.mesh.shape[2]+1))
        field[:-1,:-1,:-1] = self.field
        # copy +x
        field[-1,:-1,:-1] = self.field[0,:,:]
        # copy +y
        field[:,-1,:-1] = field[:,0,:-1]
        # copy +z
        field[:,:,-1] = field[:,:,0]

        return field

    def copy(self):
        return Field(self.mesh).from_array(self.field)

    def interpolator(self, **kwarg):
        r""" Obtain an interpolator for the field on its mesh.

        Parameters
        ----------
        \**kwarg
            Keyword arguments for :py:class:`scipy.interpolate.RegularGridInterpolator`.

        Returns
        -------
        :py:class:`scipy.interpolate.RegularGridInterpolator`
            SciPy interpolator object. The interpolator is a callable that returns values
            of the field.

        Notes
        -----
        In order to accommodate triclinic meshes, the interpolation is performed
        on the fractional coordinates. Linear interpolation is used for points that
        are off mesh.

        """
        # meshes in x, y, z w.r.t. fractions, going all the way up to 1.0
        interp_field = self.buffered()
        fx = np.arange(interp_field.shape[0]).astype(np.float32) / self.mesh.shape[0]
        fy = np.arange(interp_field.shape[1]).astype(np.float32) / self.mesh.shape[1]
        fz = np.arange(interp_field.shape[2]).astype(np.float32) / self.mesh.shape[2]

        return scipy.interpolate.RegularGridInterpolator((fx,fy,fz), interp_field, **kwarg)

class TriangulatedSurface(object):
    """ Triangulated surface mesh.

    The surface mesh is composed of *vertices* connected by edges to form
    *faces*. Each face is a triangle defined by three connected vertices.
    The surface normal to the triangle is given by the ?? rule.

    Attributes
    ----------
    vertex
    normal
    face

    """
    def __init__(self):
        self._vertex = np.empty((0,3))
        self._normal = np.empty((0,3))
        self._face = []

    def add_vertex(self, vertex, normal):
        """ Add a vertex to the surface.

        Parameters
        ----------
        vertex : array_like
            3-tuple or `N`x3 array of vertices.
        normal : array_like
            3-tuple or `N`x3 array of vertex normals.

        Raises
        ------
        IndexError
            If `vertex` and `normal` are not (arrays of) 3-element tuples of equal shape.

        """
        vertex = np.asarray(vertex)
        if len(vertex.shape) == 1:
            vertex = np.array([vertex])

        normal = np.asarray(normal)
        if len(normal.shape) == 1:
            normal = np.array([normal])

        if vertex.shape[0] != normal.shape[0]:
            raise IndexError('Must give equal number of vertexes and normals')

        if vertex.shape[1] != 3 or normal.shape[1] != 3:
            raise IndexError('Vertex and normal must be 3-element vectors')

        self._vertex = np.append(self._vertex, vertex, axis=0)
        self._normal = np.append(self._normal, normal, axis=0)

    def add_face(self, face):
        """ Add a face from vertices.

        The definition of a face should be consistent with `skimage`.

        Parameters
        ----------
        face : array_like
            3-tuple or `N`x3 array of indexes comprising a face.

        Raises
        ------
        IndexError
            If there are not 3 indices for a face.

        """
        # convert input to tuples
        face = np.asarray(face, dtype=np.int32)
        if len(face.shape) == 1:
            face = np.array([face])

        if face.shape[1] != 3:
            raise IndexError('Faces must have 3 vertices')
        face = [tuple(f.tolist()) for f in face]

        self._face.extend(face)

    @property
    def vertex(self):
        return self._vertex

    @property
    def normal(self):
        return self._normal

    @property
    def face(self):
        return tuple(self._face)
