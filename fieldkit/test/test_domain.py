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
        vol = fieldkit.domain.volume(field, threshold=0.5, N=500000, seed=42)
        self.assertAlmostEqual(vol, mesh.lattice.volume, places=1)
