""" Unit tests for mesh-related data structures.
"""
import unittest
import numpy as np
import membranekit

class MeshTest(unittest.TestCase):
    """ Test cases for :py:obj:`~membranekit.mesh.Mesh`
    """

    def test_cube(self):
        """ Test mesh formation in simple cube.
        """
        mesh = membranekit.Mesh().from_lattice(N=4,lattice=membranekit.HOOMDLattice(L=2.0))
        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.shape, (4,4,4))
        np.testing.assert_array_almost_equal(mesh.step, (0.5, 0.5, 0.5))

    def test_ortho(self):
        """ Test mesh formation in orthorhombic box.
        """
        mesh = membranekit.Mesh().from_lattice(N=(2,3,4),lattice=membranekit.HOOMDLattice(L=2.4))
        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.shape, (2,3,4))
        np.testing.assert_array_almost_equal(mesh.step, (1.2, 0.8, 0.6))

        mesh = membranekit.Mesh().from_lattice(N=(2,3,4),lattice=membranekit.HOOMDLattice(L=(0.2,0.3,0.4)))
        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.shape, (2,3,4))
        np.testing.assert_array_almost_equal(mesh.step, (0.1, 0.1, 0.1))

    def test_tilt(self):
        """ Test mesh formation with tilt factors.
        """
        mesh = membranekit.Mesh().from_lattice(N=4,lattice=membranekit.HOOMDLattice(L=4.,tilt=(0.,0.,0.)))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (3., 3., 3.))

        mesh = membranekit.Mesh().from_lattice(N=4,lattice=membranekit.HOOMDLattice(L=4.,tilt=(0.5,0.,0.)))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (4.5, 3., 3.))

        mesh = membranekit.Mesh().from_lattice(N=4,lattice=membranekit.HOOMDLattice(L=4.,tilt=(0.5,0.,0.5)))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (4.5, 4.5, 3.))

        mesh = membranekit.Mesh().from_lattice(N=4,lattice=membranekit.HOOMDLattice(L=4.,tilt=(0.5,0.5,0.5)))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (6.0, 4.5, 3.))

    def test_array(self):
        """ Test that mesh can be copied from an existing lattice.
        """
        mesh = membranekit.Mesh().from_lattice(N=(2,3,4),lattice=membranekit.HOOMDLattice(L=(0.2,0.3,0.4), tilt=(0.1,0.2,0.3)))
        mesh2 = membranekit.Mesh().from_array(mesh.grid)

        np.testing.assert_almost_equal(mesh.lattice.L, mesh2.lattice.L)
        np.testing.assert_almost_equal(mesh.lattice.matrix, mesh2.lattice.matrix)

    def test_neighbors(self):
        """ Test for determination of neighbor sites in mesh.
        """
        mesh = membranekit.Mesh().from_lattice(N=3,lattice=membranekit.HOOMDLattice(L=1))

        # point in middle of mesh
        neigh = mesh.neighbors((1,1,1))
        self.assertEqual(len(neigh), 6)
        np.testing.assert_array_equal(neigh, ((2,1,1),(0,1,1),(1,2,1),(1,0,1),(1,1,2),(1,1,0)))

        # test pbcs backward
        neigh = mesh.neighbors((0,0,0))
        self.assertEqual(len(neigh), 6)
        np.testing.assert_array_equal(neigh, ((1,0,0),(2,0,0),(0,1,0),(0,2,0),(0,0,1),(0,0,2)))

        # test pbcs forward
        neigh = mesh.neighbors((2,2,2))
        self.assertEqual(len(neigh), 6)
        np.testing.assert_array_equal(neigh, ((0,2,2),(1,2,2),(2,0,2),(2,1,2),(2,2,0),(2,2,1)))

        # half storage
        neigh = mesh.neighbors((1,1,1), full=False)
        self.assertEqual(len(neigh), 3)
        np.testing.assert_array_equal(neigh, ((2,1,1),(1,2,1),(1,1,2)))

        # test small cells
        mesh = membranekit.Mesh().from_lattice(N=2,lattice=membranekit.HOOMDLattice(L=1))
        neigh = mesh.neighbors((0,0,0))
        self.assertEqual(len(neigh), 3)
        np.testing.assert_array_equal(neigh, ((1,0,0),(0,1,0),(0,0,1)))

        neigh = mesh.neighbors((1,1,1))
        self.assertEqual(len(neigh), 3)
        np.testing.assert_array_equal(neigh, ((0,1,1),(1,0,1),(1,1,0)))

        # test stupidly small cells that can't have neighbors
        mesh = membranekit.Mesh().from_lattice(N=1,lattice=membranekit.HOOMDLattice(L=1))
        neigh = mesh.neighbors((0,0,0))
        self.assertEqual(len(neigh),0)

class FieldTest(unittest.TestCase):
    """ Test cases for :py:obj:`~membranekit.mesh.Field`.
    """
    def setUp(self):
        self.mesh = membranekit.Mesh().from_lattice(N=(2,3,4),lattice=membranekit.HOOMDLattice(L=2.0))

    def test(self):
        """ Test for basic creation of an empty field.
        """
        field = membranekit.Field(self.mesh)
        np.testing.assert_almost_equal(field.field, np.zeros(self.mesh.shape))
        self.assertAlmostEqual(field[0,0,0], 0.)
        self.assertEqual(field.shape, self.mesh.shape)

        # change field data via property
        field.field = np.ones(self.mesh.shape)
        np.testing.assert_almost_equal(field.field, np.ones(self.mesh.shape))
        self.assertAlmostEqual(field[0,0,0], 1.)

    def test_from_array(self):
        """ Test for field creation for a simple array.
        """
        data = np.ones(self.mesh.shape)
        field = membranekit.Field(self.mesh).from_array(data)
        self.assertEqual(field.shape, self.mesh.shape)
        np.testing.assert_almost_equal(field.field, np.ones(self.mesh.shape))

    def test_multi(self):
        """ Test common multidimensional layouts for input arrays.
        """
        # last index is density
        data = np.ones(np.append(self.mesh.shape, 2))
        field = membranekit.Field(self.mesh).from_array(data, index=0, axis=3)
        self.assertEqual(field.shape, self.mesh.shape)
        np.testing.assert_almost_equal(field.field, np.ones(self.mesh.shape))

        # first index is density
        data = np.ones(np.insert(self.mesh.shape, 0, 2))
        data[1] *= 2.
        field = membranekit.Field(self.mesh).from_array(data, index=1, axis=0)
        self.assertEqual(field.shape, self.mesh.shape)
        np.testing.assert_almost_equal(field.field, np.full(self.mesh.shape,2.))

    def test_interpolator(self):
        """ Test for creation of interpolator into field.
        """
        field = membranekit.Field(self.mesh)
        for n in np.ndindex(self.mesh.shape):
            field[n] = n[-2]
        np.testing.assert_almost_equal(field.field, ( ( (0,0,0,0),(1,1,1,1),(2,2,2,2) ),
                                                      ( (0,0,0,0),(1,1,1,1),(2,2,2,2) ) ) )

        # interpolator
        f = field.interpolator()

        # interpolate the mesh points themselves, which should have values almost the same as the field
        pts = self.mesh.grid.reshape((np.prod(self.mesh.shape),3))
        fracs = self.mesh.lattice.as_fraction(pts)
        vals = f(fracs).reshape(self.mesh.shape)
        np.testing.assert_almost_equal(vals,  ( ( (0,0,0,0),(1,1,1,1),(2,2,2,2) ),
                                                ( (0,0,0,0),(1,1,1,1),(2,2,2,2) ) ) )

        # interpolate in between the mesh. last point gets 1 since interpolation is between 2 and 0
        pts += 0.5 * self.mesh.step
        fracs = self.mesh.lattice.as_fraction(pts)
        vals = f(fracs).reshape(self.mesh.shape)
        np.testing.assert_almost_equal(vals,  ( ( (0.5,0.5,0.5,0.5),(1.5,1.5,1.5,1.5),(1.0,1.0,1.0,1.0) ),
                                                ( (0.5,0.5,0.5,0.5),(1.5,1.5,1.5,1.5),(1.0,1.0,1.0,1.0) ) ) )
