""" Unit tests for domain analysis.

"""
import unittest
import numpy as np
import fieldkit

class DomainTest(unittest.TestCase):
    """ Tests for domain characterization methods.
    """
    def setUp(self):
        self.mesh = fieldkit.Mesh().from_lattice(N=4, lattice=fieldkit.HOOMDLattice(L=4.0))
        self.field = fieldkit.Field(self.mesh).from_array(np.zeros(self.mesh.shape))

    def test_digitize(self):
        """ Test for digitization of a sample by a threshold.
        """
        self.field[:,:,0] = 1.
        self.field[:,:,2] = 1.

        domain = fieldkit.domain.digitize(self.field, 0.5)
        self.assertEqual(len(domain.nodes), 4*4*2)
        # reference is all of the nodes having k = 0 or 2
        ref = self.mesh.indices[np.logical_or(self.mesh.indices[:,:,:,2] == 0, self.mesh.indices[:,:,:,2] == 2)]
        np.testing.assert_array_equal(domain.nodes, ref)

    def test_find(self):
        """ Test for domain detection by :py:meth:`~fieldkit.domain.find`.
        """
        self.field[:,:,0] = 1.
        self.field[:,:,2] = 1.

        # check that two domains are identified with the correct field points
        domains = fieldkit.domain.find(fieldkit.domain.digitize(self.field,0.5))
        domains = list(domains)
        self.assertEqual(len(domains),2)
        for d,col in zip((0,1),(0,2)):
            pts = []
            for n in sorted(domains[d].nodes):
                pts.append(self.mesh[n])
            pts = np.array(pts)
            test = self.mesh[:,:,col].reshape(16,3)
            np.testing.assert_almost_equal(pts, test)

        # join fields through the boundary
        self.field[:,:,3] = 1.
        domains = fieldkit.domain.find(fieldkit.domain.digitize(self.field,0.5))
        domains = list(domains)
        self.assertEqual(len(domains),1)

        # bridge everything through another dimension
        self.field[:,:,3] = 0.
        self.field[1,:,:] = 1.
        domains = fieldkit.domain.find(fieldkit.domain.digitize(self.field,0.5))
        domains = list(domains)
        self.assertEqual(len(domains),1)

        # increase the tolerance so that nothing is in a domain
        domains = fieldkit.domain.find(fieldkit.domain.digitize(self.field,1.1))
        domains = list(domains)
        self.assertEqual(len(domains),0)

    def test_is_connected(self):
        """ Test for connectivity of a domain.
        """
        self.field[:,:,0] = 1.
        self.field[:,:,1] = 0.4
        self.field[:,:,2] = 1.

        # domain is uninitally unconnected
        domain = fieldkit.domain.digitize(self.field, 0.5)
        self.assertFalse(fieldkit.domain.is_connected(domain))

        # domain is connected now
        domain = fieldkit.domain.digitize(self.field, 0.3)
        self.assertTrue(fieldkit.domain.is_connected(domain))

    def test_is_percolated(self):
        """ Test for percolation of a domain.
        """
        # percolate x
        self.field[:,0,0] = 1.
        domain = fieldkit.domain.digitize(self.field, 0.5)
        self.assertTrue(fieldkit.domain.is_percolated(domain, axis=0))
        self.assertFalse(fieldkit.domain.is_percolated(domain, axis=1))
        self.assertFalse(fieldkit.domain.is_percolated(domain, axis=2))

        # percolate x/y
        self.field[0,:,0] = 1.
        domain = fieldkit.domain.digitize(self.field, 0.5)
        self.assertTrue(fieldkit.domain.is_percolated(domain, axis=0))
        self.assertTrue(fieldkit.domain.is_percolated(domain, axis=1))
        self.assertFalse(fieldkit.domain.is_percolated(domain, axis=2))

        # percolate x/y/z
        self.field[0,0,:] = 1.
        domain = fieldkit.domain.digitize(self.field, 0.5)
        self.assertTrue(fieldkit.domain.is_percolated(domain, axis=0))
        self.assertTrue(fieldkit.domain.is_percolated(domain, axis=1))
        self.assertTrue(fieldkit.domain.is_percolated(domain, axis=2))

    def test_tortuosity(self):
        """ Test for tortuosity calculation for a domain.
        """
        ## one straight line
        self.field[:,0,0] = 1.
        domain = fieldkit.domain.digitize(self.field, 0.5)
        # x
        tort,nodes = fieldkit.domain.tortuosity(domain, axis=0)
        self.assertAlmostEqual(tort[0], 1.0)
        self.assertEqual(nodes[0], (0,0,0))
        # y
        tort,nodes = fieldkit.domain.tortuosity(domain, axis=1)
        self.assertEqual(len(tort),0)
        # z
        tort,nodes = fieldkit.domain.tortuosity(domain, axis=1)
        self.assertEqual(len(tort),0)

        ## three lines
        self.field[:,0,0] = 1.
        self.field[:,2,1] = 1.
        self.field[:,1,2] = 1.
        tort,nodes = fieldkit.domain.tortuosity(fieldkit.domain.digitize(self.field, 0.5), axis=0)
        np.testing.assert_array_equal(nodes,((0,0,0),(0,1,2),(0,2,1)))
        np.testing.assert_almost_equal(tort, (1.0, 1.0, 1.0))

        ## single plane
        self.field[:] = 0.
        self.field[:,:,0] = 1.
        domain = fieldkit.domain.digitize(self.field, 0.5)
        # x
        tort,nodes = fieldkit.domain.tortuosity(domain, axis=0)
        self.assertEqual(len(tort),4)
        np.testing.assert_array_equal(nodes, ((0,0,0),(0,1,0),(0,2,0),(0,3,0)))
        np.testing.assert_almost_equal(tort, (1.0,1.0,1.0,1.0))
        # y
        tort,nodes = fieldkit.domain.tortuosity(domain, axis=1)
        self.assertEqual(len(tort),4)
        np.testing.assert_array_equal(nodes, ((0,0,0),(1,0,0),(2,0,0),(3,0,0)))
        np.testing.assert_almost_equal(tort, (1.0,1.0,1.0,1.0))
        # z
        tort,nodes = fieldkit.domain.tortuosity(domain, axis=2)
        self.assertEqual(len(tort),0)

        ## twisted line
        #
        # Shape::
        #   --
        # -|  |-
        #
        # Total length = 6, tortuosity = 6/4 = 1.5
        self.field[:] = 0.
        for pt in ((0,0,0),(1,0,0),(1,1,0),(2,1,0),(3,1,0),(3,0,0)):
            self.field[pt] = 1.
        domain = fieldkit.domain.digitize(self.field, 0.5)
        tort,nodes = fieldkit.domain.tortuosity(domain, axis=0)
        self.assertEqual(len(tort),1)
        self.assertAlmostEqual(tort[0], 1.5)

        ## triclinic mesh with 45* tilt
        tri_mesh = fieldkit.Mesh().from_lattice(N=4, lattice=fieldkit.HOOMDLattice(L=4.0, tilt=(1.0,0.,0.)))
        tri_field = fieldkit.Field(tri_mesh).from_array(np.zeros(tri_mesh.shape))
        # x: not tilted, so tortuosity of a line is 1
        tri_field[:,0,0] = 1.
        domain = fieldkit.domain.digitize(tri_field, 0.5)
        tort,nodes = fieldkit.domain.tortuosity(domain, axis=0)
        self.assertAlmostEqual(tort[0], 1.0)
        # y: tilted by 45*, so tortuosity is sqrt(2) to account for longer path
        tri_field[:] = 0.
        tri_field[0,:,0] = 1.
        domain = fieldkit.domain.digitize(tri_field, 0.5)
        tort,nodes = fieldkit.domain.tortuosity(domain, axis=1)
        self.assertAlmostEqual(tort[0], np.sqrt(2.))

class DomainBurnTest(unittest.TestCase):
    """ Tests for burning algorithm
    """
    def test_plane(self):
        mesh = fieldkit.Mesh().from_lattice(N=4, lattice=fieldkit.HOOMDLattice(L=4.))
        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        field[:,:,0] = 0
        domain = fieldkit.domain.digitize(field, 0.5)

        burn,axis = fieldkit.domain.burn(domain)
        # check burning points
        np.testing.assert_equal(burn.shape, (4,4,4))
        np.testing.assert_equal(burn[:,:,0], 0)
        np.testing.assert_equal(burn[:,:,1], 1)
        np.testing.assert_equal(burn[:,:,2], 2)
        np.testing.assert_equal(burn[:,:,3], 1)
        # check axis
        axis_nodes = np.asarray(axis.nodes)
        np.testing.assert_equal(axis_nodes.shape, (4*4,3))
        np.testing.assert_equal(axis_nodes[:,2], 2)

    def test_collide(self):
        mesh = fieldkit.Mesh().from_lattice(N=3, lattice=fieldkit.HOOMDLattice(L=4.))
        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        field[:,:,0] = 0
        domain = fieldkit.domain.digitize(field, 0.5)

        burn,axis = fieldkit.domain.burn(domain)
        # check burning points
        np.testing.assert_equal(burn.shape, (3,3,3))
        np.testing.assert_equal(burn[:,:,0], 0)
        np.testing.assert_equal(burn[:,:,1], 1)
        np.testing.assert_equal(burn[:,:,2], 1)
        # check axis
        axis_nodes = np.asarray(axis.nodes)
        np.testing.assert_equal(axis_nodes.shape, (2*3*3,3))
        np.testing.assert_equal(axis_nodes[:,2], [1,2]*9)
