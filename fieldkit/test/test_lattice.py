""" Unit tests for lattice data structures.
"""
import unittest
import numpy as np
import fieldkit

class LatticeTest(unittest.TestCase):
    """ Test cases for :py:obj:`~fieldkit.lattice.Lattice` and :py:obj:`~fieldkit.lattice.HOOMDLattice`.
    """

    def test(self):
        """ Test for basic properties of a skewed lattice.
        """
        lattice = fieldkit.Lattice([1,0,0],[0,2,2],[0,0,3])

        # size of lattice
        np.testing.assert_almost_equal(lattice.L, [1,np.sqrt(8),3])

        # lattice vectors
        np.testing.assert_almost_equal(lattice.a, [1,0,0])
        np.testing.assert_almost_equal(lattice.b, [0,2,2])
        np.testing.assert_almost_equal(lattice.c, [0,0,3])

        # cell matrix
        np.testing.assert_almost_equal(lattice.matrix, ((1,0,0),(0,2,0),(0,2,3)))

        # inverse cell matrix
        np.testing.assert_almost_equal(lattice.inverse, ((1,0,0),(0,0.5,0),(0,-1./3.,1./3.)))

        # volume
        self.assertAlmostEqual(lattice.volume, 6)

    def test_hoomd(self):
        """ Test for creation of various lattices in the hoomd / lammps definition.
        """
        # cube
        lattice = fieldkit.HOOMDLattice(L=2)
        np.testing.assert_almost_equal(lattice.L, (2.0,2.0,2.0))
        np.testing.assert_almost_equal(np.diag(lattice.matrix), (2.0,2.0,2.0))
        np.testing.assert_almost_equal(lattice.matrix-np.diag(lattice.L), np.zeros((3,3)))
        self.assertAlmostEqual(lattice.volume, 2**3)

        # orthorhombic
        lattice = fieldkit.HOOMDLattice(L=(0.2,0.3,0.4))
        np.testing.assert_almost_equal(lattice.L, (0.2,0.3,0.4))
        np.testing.assert_almost_equal(np.diag(lattice.matrix), (0.2,0.3,0.4))
        np.testing.assert_almost_equal(lattice.matrix-np.diag(lattice.L), np.zeros((3,3)))
        self.assertAlmostEqual(lattice.volume, 0.2*0.3*0.4)

        # tilted in xy
        lattice = fieldkit.HOOMDLattice(L=4.,tilt=(0.5,0.,0.))
        np.testing.assert_almost_equal(lattice.matrix, ((4.,2.,0.),(0.,4.,0.),(0.,0.,4.)))
        self.assertAlmostEqual(lattice.volume, 4**3)

        # tilted in xy and yz
        lattice = fieldkit.HOOMDLattice(L=4.,tilt=(0.5,0.,0.5))
        np.testing.assert_almost_equal(lattice.matrix, ((4.,2.,0.),(0.,4.,2.),(0.,0.,4.)))
        self.assertAlmostEqual(lattice.volume, 4**3)

        # tilted in xy, xz, and yz
        lattice = fieldkit.HOOMDLattice(L=4.,tilt=(0.5,0.5,0.5))
        np.testing.assert_almost_equal(lattice.matrix, ((4.,2.,2.),(0.,4.,2.),(0.,0.,4.)))
        self.assertAlmostEqual(lattice.volume, 4**3)

    def test_coordinate(self):
        """ Test for mapping of fractional coordinates to real coordinates.
        """
        lattice = fieldkit.HOOMDLattice(L=(1,2,4))
        r = lattice.as_coordinate((0.5, 0.25, 0.125))
        np.testing.assert_array_almost_equal(r, (0.5, 0.5, 0.5))

        # two at once
        r = lattice.as_coordinate(((0., 0., 0.),(1.0,1.0,1.0)))
        np.testing.assert_array_almost_equal(r, ((0,0,0),(1,2,4)))

        lattice = fieldkit.HOOMDLattice(L=4, tilt=(0.5,0.,0.))
        r = lattice.as_coordinate((0.5, 0.5, 0.5))
        np.testing.assert_array_almost_equal(r, (3., 2., 2.))

        lattice = fieldkit.HOOMDLattice(L=4, tilt=(0.5,0.,0.5))
        r = lattice.as_coordinate((0.5, 0.5, 0.5))
        np.testing.assert_array_almost_equal(r, (3., 3., 2.))

    def test_fraction(self):
        """ Test for mapping of real coordinates to fractional coordinates.
        """
        lattice = fieldkit.HOOMDLattice(L=(1,2,4))
        f = lattice.as_fraction((0.5,0.5,0.5))
        np.testing.assert_almost_equal(f, (0.5, 0.25, 0.125))

        # two at once
        f = lattice.as_fraction(((0,0,0),(1, 2, 4)))
        np.testing.assert_array_almost_equal(f, ((0.,0.,0.),(1.,1.,1.)))

        lattice = fieldkit.HOOMDLattice(L=4, tilt=(0.5,0.,0.))
        f = lattice.as_fraction((3.,2.,2.))
        np.testing.assert_almost_equal(f, (0.5, 0.5, 0.5))

        lattice = fieldkit.HOOMDLattice(L=4, tilt=(0.5,0.,0.5))
        f = lattice.as_fraction((3.,3.,2.))
        np.testing.assert_almost_equal(f, (0.5, 0.5, 0.5))

    def test_orthorhombic(self):
        """ Test for construction of orthorhombic basis from triclinic lattice.
        """
        tri = fieldkit.HOOMDLattice(L=(2.,3.,4.), tilt=(0.5, 0.5, 0.5))
        ortho = tri.to_orthorhombic()

        self.assertAlmostEqual(ortho.volume, tri.volume)
        np.testing.assert_almost_equal(ortho.a, (2,0,0))
        np.testing.assert_almost_equal(ortho.b, (0,3,0))
        np.testing.assert_almost_equal(ortho.c, (0,0,4))
