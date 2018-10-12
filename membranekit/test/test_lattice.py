""" Unit tests for lattice data structures.

"""
import unittest
import numpy as np
import membranekit

class LatticeTest(unittest.TestCase):
    def test(self):
        lattice = membranekit.Lattice([1,0,0],[0,2,2],[0,0,3])

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

    def test_hoomd(self):
        # cube
        lattice = membranekit.HOOMDLattice(L=2)
        np.testing.assert_almost_equal(lattice.L, (2.0,2.0,2.0))
        np.testing.assert_almost_equal(np.diag(lattice.matrix), (2.0,2.0,2.0))
        np.testing.assert_almost_equal(lattice.matrix-np.diag(lattice.L), np.zeros((3,3)))

        # orthorhombic
        lattice = membranekit.HOOMDLattice(L=(0.2,0.3,0.4))
        np.testing.assert_almost_equal(lattice.L, (0.2,0.3,0.4))
        np.testing.assert_almost_equal(np.diag(lattice.matrix), (0.2,0.3,0.4))
        np.testing.assert_almost_equal(lattice.matrix-np.diag(lattice.L), np.zeros((3,3)))

        # tilted in xy
        lattice = membranekit.HOOMDLattice(L=4.,tilt=(0.5,0.,0.))
        np.testing.assert_almost_equal(lattice.matrix, ((4.,2.,0.),(0.,4.,0.),(0.,0.,4.)))

        # tilted in xy and yz
        lattice = membranekit.HOOMDLattice(L=4.,tilt=(0.5,0.,0.5))
        np.testing.assert_almost_equal(lattice.matrix, ((4.,2.,0.),(0.,4.,2.),(0.,0.,4.)))

        # tilted in xy, xz, and yz
        lattice = membranekit.HOOMDLattice(L=4.,tilt=(0.5,0.5,0.5))
        np.testing.assert_almost_equal(lattice.matrix, ((4.,2.,2.),(0.,4.,2.),(0.,0.,4.)))

    def test_coordinate(self):
        raise NotImplementedError()

    def test_fraction(self):
        lattice = membranekit.HOOMDLattice(L=(1,2,4))
        f = lattice.as_fraction((0.5,0.5,0.5))
        np.testing.assert_almost_equal(f, (0.5, 0.25, 0.125))

        lattice = membranekit.HOOMDLattice(L=4, tilt=(0.5,0.,0.))
        f = lattice.as_fraction((3.,2.,2.))
        np.testing.assert_almost_equal(f, (0.5, 0.5, 0.5))

        lattice = membranekit.HOOMDLattice(L=4, tilt=(0.5,0.,0.5))
        f = lattice.as_fraction((3.,3.,2.))
        np.testing.assert_almost_equal(f, (0.5, 0.5, 0.5))

    def test_orthorhombic(self):
        raise NotImplementedError()
