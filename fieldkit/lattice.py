""" Data structures for working with lattices.

"""
from __future__ import division
import numpy as np

__all__ = ["Lattice","HOOMDLattice"]

class Lattice(object):
    """ Lattice representing the three-dimensional periodic cell

    Parameters
    ----------
    a : array_like
        First lattice vector.
    b : array_like
        Second lattice vector.
    c : array_like
        Third lattice vector.

    Attributes
    ----------
    a
    b
    c
    matrix
    inverse
    L
    volume

    """
    def __init__(self, a, b, c):
        self._matrix = np.column_stack((a,b,c))
        self._inverse = np.linalg.inv(self.matrix)
        self._L = np.linalg.norm(self.matrix,axis=0)
        self._volume = np.dot(np.cross(a,b),c)

    @property
    def a(self):
        return self._matrix[:,0]

    @property
    def b(self):
        return self._matrix[:,1]

    @property
    def c(self):
        return self._matrix[:,2]

    @property
    def matrix(self):
        """ Periodic cell matrix.

        Each column of the matrix corresponds to a lattice vector.::

            [a, b, c]

        Returns
        -------
        array_like:
            3x3 matrix containing the lattice vectors.

        """
        return self._matrix

    @property
    def inverse(self):
        """ Inverse of the cell matrix.

        The inverse matrix for the cell is useful for projecting
        Cartesian coordinates onto the lattice vectors.

        Returns
        -------
        array_like:
            The 3x3 inverse of :py:attr:`~matrix`.

        """
        return self._inverse

    @property
    def L(self):
        """ Lengths of the lattice vectors.

        Returns
        -------
        array_like:
            Lengths of each lattice vector.

        """
        return self._L

    @property
    def volume(self):
        """ Volume of the lattice.

        Returns
        -------
        float:
            The volume enclosed by the parallelopiped defining the lattice.

        Notes
        -----
        The volume is precomputed from the lattice vectors using the scalar triple product.

        .. math::

            V = (a \times b) \cdot c

        """
        return self._volume

    def as_coordinate(self, f):
        """ Convert a fractional value to a coordinate from the lattice vectors.

        Parameters
        ----------
        f : array_like
            `N`x3 array of fractional coordinates.

        Returns
        -------
        array_like
            `N`x3 array of real coordinates.

        """
        f = np.asarray(f)
        return np.dot(f, self.matrix.transpose())

    def as_fraction(self, r):
        """ Convert a point into fractional coordinates from the lattice vectors.

        Parameters
        ----------
        f : array_like
            `N`x3 array of real coordinates.

        Returns
        -------
        array_like
            `N`x3 array of fractional coordinates.

        """

        r = np.asarray(r)
        return np.dot(r, self.inverse.transpose())

    def to_orthorhombic(self):
        """ Project lattice onto an orthorhombic basis.

        Returns
        -------
        :py:obj:`~Lattice`
            A representation of the current lattice in an orthorhombic basis.

        Notes
        -----
        The orthorhombic basis is constructed using the Gram-Schmidt method.
        The lattice vectors **a**, **b**, and **c** are iteratively projected
        to form a basis. The initial lattice vectors are then projected onto
        the basis to determine their sizes.

        """
        # lattice vectors
        a = self.matrix[:,0]
        b = self.matrix[:,1]
        c = self.matrix[:,2]

        # find orthogonal basis vectors for the lattice using Gram-Schmidt method
        # projection of u onto v
        proj = lambda u,v : np.dot(u,v)/np.dot(v,v)*v
        v1 = a
        v2 = b - proj(b,v1)
        v3 = c - proj(c,v1) - proj(c,v2)

        # v1, v2, v3 now form an orthogonal basis, but we need to size the lattice
        # by projecting onto the new coordinates
        a_new = proj(a, v1)
        b_new = proj(b, v2)
        c_new = proj(c, v3)

        return Lattice(a_new, b_new, c_new)

class HOOMDLattice(Lattice):
    def __init__(self, L, tilt=None):
        """ Lattice consistent with HOOMD-blue / LAMMPS geometries.

        HOOMD-blue and LAMMPS use a triclinic cell with a specific
        orientation. This class provides a convenient way to build
        the lattice vectors from those parameters.

        Parameters
        ----------
        L : float or array_like
            Length of orthorhombic (undeformed) box edges. If only
            one value is specified, the undeformed cell is cubic.
        tilt : None or array_like
            Fractional tilt factors for *xy*, *xz*, and *yz*.

        Notes
        -----
        The HOOMD-blue tilt factors can be converted to LAMMPS
        tilt factors by multiplying `tilt` by the `L` associated
        with the second index of each factor.

        """
        L = np.asarray(L)
        try:
            if len(L) != 3:
               raise IndexError('HOOMDLattice is 3D')
        except TypeError:
            L = np.full(3,L)

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
        a = L[0] * np.array((1., 0., 0.))
        b = L[1] * np.array((tilt[0], 1., 0.))
        c = L[2] * np.array((tilt[1],tilt[2],1.))

        # use parent lattice constructor with these vectors
        try:
            super().__init__(a,b,c)
        except:
            super(Lattice, self).__init__(a,b,c)
