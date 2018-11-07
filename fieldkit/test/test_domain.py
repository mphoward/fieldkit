""" Unit tests for domain analysis.

"""
import unittest
import numpy as np
import fieldkit

class DomainTest(unittest.TestCase):
    def test_find(self):
        """ Test for domain detection by :py:meth:`~fieldkit.domain.find`.
        """
        mesh = fieldkit.Mesh().from_lattice(N=4, lattice=fieldkit.HOOMDLattice(L=4.0))
        field = fieldkit.Field(mesh).from_array(np.zeros(mesh.shape))
        field[:,:,0] = 1.
        field[:,:,2] = 1.

        # check that two domains are identified with the correct field points
        domains = fieldkit.domain.find(field, 0.5)
        domains = list(domains)
        self.assertEqual(len(domains),2)
        for d,col in zip((0,1),(0,2)):
            pts = []
            for n in sorted(domains[d]):
                pts.append(mesh[n])
            pts = np.array(pts)
            test = mesh[:,:,col].reshape(16,3)
            np.testing.assert_almost_equal(pts, test)

        # join fields through the boundary
        field[:,:,3] = 1.
        domains = fieldkit.domain.find(field, 0.5)
        domains = list(domains)
        self.assertEqual(len(domains),1)

        # bridge everything through another dimension
        field[:,:,3] = 0.
        field[1,:,:] = 1.
        domains = fieldkit.domain.find(field, 0.5)
        domains = list(domains)
        self.assertEqual(len(domains),1)

        # increase the tolerance so that nothing is in a domain
        domains = fieldkit.domain.find(field, 1.1)
        domains = list(domains)
        self.assertEqual(len(domains),0)

    def test_volume(self):
        """ Test for domain volume calculation by :py:meth:`~fieldkit.domain.volume`.
        """
        mesh = fieldkit.Mesh().from_lattice(N=4, lattice=fieldkit.HOOMDLattice(L=4.0))
        self.assertAlmostEqual(mesh.lattice.volume, 64.)

        # 25% covered
        field = fieldkit.Field(mesh).from_array(np.zeros(mesh.shape))
        field[:,:,0] = 1.
        vol = fieldkit.domain.volume(field, threshold=0.5, N=500000, seed=42)
        self.assertAlmostEqual(vol, 0.25*mesh.lattice.volume, places=1)

        # 50% covered
        field[:,:,2] = 1.
        vol = fieldkit.domain.volume(field, threshold=0.5, N=500000, seed=42)
        self.assertAlmostEqual(vol, 0.5*mesh.lattice.volume, places=1)

        # 75% covered
        field[:,:,1] = 1.
        vol = fieldkit.domain.volume(field, threshold=0.5, N=500000, seed=42)
        self.assertAlmostEqual(vol, 0.75*mesh.lattice.volume, places=1)

        # 100% covered
        field[:,:,3] = 1.
        vol = fieldkit.domain.volume(field, threshold=0.5, N=5e5, seed=42)
        self.assertAlmostEqual(vol, mesh.lattice.volume, places=1)

    def test_area(self):
        """ Test for domain surface triangulation and area calculation.
        """
        mesh = fieldkit.Mesh().from_lattice(N=4, lattice=fieldkit.HOOMDLattice(L=4.0))
        self.assertAlmostEqual(mesh.lattice.volume, 64.)

        # make a plane in the box
        field = fieldkit.Field(mesh).from_array(np.zeros(mesh.shape))
        field[:,:,0] = 1.
        surface = fieldkit.domain.triangulate(field, threshold=0.5)
        area = fieldkit.domain.surface_area(surface)

        self.assertAlmostEqual(area, 2*mesh.lattice.L[0]*mesh.lattice.L[1])

    def test_sphere(self):
        """ Test for measuring properties of a sphere
        """
        mesh = fieldkit.Mesh().from_lattice(N=32, lattice=fieldkit.HOOMDLattice(L=4.))
        field = fieldkit.Field(mesh).from_array(np.zeros(mesh.shape))

        # make a sphere
        R = 1.
        for n in np.ndindex(mesh.shape):
            pt = mesh[n]
            rsq = np.sum((pt-mesh.lattice.L/2)**2)
            if rsq <= R**2:
                field[n] = 1.

        # use a loose tolerance due to inaccuracies of meshing and interpolating densities
        volume = fieldkit.domain.volume(field, threshold=0.5, N=5e5, seed=42)
        self.assertAlmostEqual(volume, 4*np.pi*R**3/3, delta=0.1)

        # the surface should have a measured area greater than that of sphere
        surface = fieldkit.domain.triangulate(field, threshold=0.5)
        area = fieldkit.domain.surface_area(surface)
        self.assertTrue(area >= 4*np.pi*R**2)
        self.assertAlmostEqual(area, 4*np.pi*R**2, delta=1.)

class DomainBurnTest(unittest.TestCase):
    """ Tests for burning algorithm
    """
    def test_plane(self):
        mesh = fieldkit.Mesh().from_lattice(N=4, lattice=fieldkit.HOOMDLattice(L=4.))
        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        field[:,:,0] = 0

        burn,axis = fieldkit.domain.burn(field, 0.5)
        # check burning points
        np.testing.assert_equal(burn.shape, (4,4,4))
        np.testing.assert_equal(burn[:,:,0], 0)
        np.testing.assert_equal(burn[:,:,1], 1)
        np.testing.assert_equal(burn[:,:,2], 2)
        np.testing.assert_equal(burn[:,:,3], 1)
        # check axis
        np.testing.assert_equal(axis.shape, (4*4,3))
        np.testing.assert_equal(axis[:,2], 2)

    def test_collide(self):
        mesh = fieldkit.Mesh().from_lattice(N=3, lattice=fieldkit.HOOMDLattice(L=4.))
        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        field[:,:,0] = 0

        burn,axis = fieldkit.domain.burn(field, 0.5)
        # check burning points
        np.testing.assert_equal(burn.shape, (3,3,3))
        np.testing.assert_equal(burn[:,:,0], 0)
        np.testing.assert_equal(burn[:,:,1], 1)
        np.testing.assert_equal(burn[:,:,2], 1)
        # check axis
        np.testing.assert_equal(axis.shape, (2*3*3,3))
        np.testing.assert_equal(axis[:,2], [1,2]*9)
