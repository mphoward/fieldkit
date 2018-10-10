""" Unit tests for data structures.

"""
import unittest
import numpy as np
import membranekit

class MeshTest(unittest.TestCase):
    """ Test cases for :py:obj:`~membranekit.data.Mesh`
    """

    def test_cube(self):
        """ Test mesh formation in simple cube.
        """
        mesh = membranekit.Mesh().from_lattice(N=4,L=2.0)
        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.shape, (4,4,4))
        np.testing.assert_almost_equal(mesh.L, (2.0,2.0,2.0))
        np.testing.assert_almost_equal(mesh.tilt, (0., 0., 0.))

    def test_ortho(self):
        """ Test mesh formation in orthorhombic box.
        """
        mesh = membranekit.Mesh().from_lattice(N=(2,3,4),L=2.4)
        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.shape, (2,3,4))
        np.testing.assert_almost_equal(mesh.L, (2.4,2.4,2.4))
        np.testing.assert_almost_equal(mesh.tilt, (0., 0., 0.))

        mesh = membranekit.Mesh().from_lattice(N=(2,3,4),L=(0.2,0.3,0.4))
        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.shape, (2,3,4))
        np.testing.assert_almost_equal(mesh.L, (0.2,0.3,0.4))
        np.testing.assert_almost_equal(mesh.tilt, (0., 0., 0.))

    def test_tilt(self):
        """ Test mesh formation with tilt factors.
        """
        mesh = membranekit.Mesh().from_lattice(N=4,L=4.,tilt=(0.,0.,0.))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (3., 3., 3.))

        mesh = membranekit.Mesh().from_lattice(N=4,L=4.,tilt=(0.5,0.,0.))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (4.5, 3., 3.))

        mesh = membranekit.Mesh().from_lattice(N=4,L=4.,tilt=(0.5,0.,0.5))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (4.5, 4.5, 3.))

        mesh = membranekit.Mesh().from_lattice(N=4,L=4.,tilt=(0.5,0.5,0.5))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (6.0, 4.5, 3.))

    def test_array(self):
        """ Test that mesh can be copied from an existing lattice.
        """
        mesh = membranekit.Mesh().from_lattice(N=(2,3,4),L=(0.2,0.3,0.4), tilt=(0.1,0.2,0.3))
        mesh2 = membranekit.Mesh().from_array(mesh.grid)

        np.testing.assert_almost_equal(mesh.L, mesh2.L)
        np.testing.assert_almost_equal(mesh.tilt, mesh2.tilt)
