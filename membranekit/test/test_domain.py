""" Unit tests for domain analysis.

"""
import unittest
import numpy as np
import membranekit

class DomainTest(unittest.TestCase):
    def test_find(self):
        mesh = membranekit.Mesh().from_lattice(N=4, lattice=membranekit.HOOMDLattice(L=4.0))

        field = membranekit.Field(mesh).from_array(np.zeros(mesh.shape))
        field[:,:,0] = 1.
        field[:,:,2] = 1.

        # check that two domains are identified with the correct field points
        domains = membranekit.domain.find(field, 0.5)
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
        domains = membranekit.domain.find(field, 0.5)
        domains = list(domains)
        self.assertEqual(len(domains),1)

        # bridge everything through another dimension
        field[:,:,3] = 0.
        field[1,:,:] = 1.
        domains = membranekit.domain.find(field, 0.5)
        domains = list(domains)
        self.assertEqual(len(domains),1)

        # increase the tolerance so that nothing is in a domain
        domains = membranekit.domain.find(field, 1.1)
        domains = list(domains)
        self.assertEqual(len(domains),0)
