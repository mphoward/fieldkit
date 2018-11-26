""" Unit tests for fieldkit.measure.

"""
import unittest
import numpy as np
import fieldkit

class MeasureTest(unittest.TestCase):
    def test_volume(self):
        """ Test for domain volume calculation by :py:meth:`~fieldkit.domain.volume`.
        """
        mesh = fieldkit.Mesh().from_lattice(N=4, lattice=fieldkit.HOOMDLattice(L=4.0))
        self.assertAlmostEqual(mesh.lattice.volume, 64.)

        # 25% covered
        field = fieldkit.Field(mesh).from_array(np.zeros(mesh.shape))
        field[:,:,0] = 1.
        vol = fieldkit.measure.volume(field, threshold=0.5, N=500000, seed=42)
        self.assertAlmostEqual(vol, 0.25*mesh.lattice.volume, places=1)

        # 50% covered
        field[:,:,2] = 1.
        vol = fieldkit.measure.volume(field, threshold=0.5, N=500000, seed=42)
        self.assertAlmostEqual(vol, 0.5*mesh.lattice.volume, places=1)

        # 75% covered
        field[:,:,1] = 1.
        vol = fieldkit.measure.volume(field, threshold=0.5, N=500000, seed=42)
        self.assertAlmostEqual(vol, 0.75*mesh.lattice.volume, places=1)

        # 100% covered
        field[:,:,3] = 1.
        vol = fieldkit.measure.volume(field, threshold=0.5, N=5e5, seed=42)
        self.assertAlmostEqual(vol, mesh.lattice.volume, places=1)

    def test_area(self):
        """ Test for domain surface triangulation and area calculation.
        """
        mesh = fieldkit.Mesh().from_lattice(N=4, lattice=fieldkit.HOOMDLattice(L=4.0))
        self.assertAlmostEqual(mesh.lattice.volume, 64.)

        # make a plane in the box
        field = fieldkit.Field(mesh).from_array(np.zeros(mesh.shape))
        field[:,:,0] = 1.
        surface = fieldkit.measure.triangulate(field, threshold=0.5)
        area = fieldkit.measure.surface_area(surface)

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
        volume = fieldkit.measure.volume(field, threshold=0.5, N=5e5, seed=42)
        self.assertAlmostEqual(volume, 4*np.pi*R**3/3, delta=0.1)

        # the surface should have a measured area greater than that of sphere
        surface = fieldkit.measure.triangulate(field, threshold=0.5)
        area = fieldkit.measure.surface_area(surface)
        self.assertTrue(area >= 4*np.pi*R**2)
        self.assertAlmostEqual(area, 4*np.pi*R**2, delta=1.)
