""" Unit tests for fieldkit.simulate

"""
import unittest
import numpy as np
import fieldkit

class RandomWalkTest(unittest.TestCase):
    """ Test cases for :py:class:`~fieldkit.simulate.random_walk`
    """

    def test_one_step(self):
        """ Test simple random walk rules for one step.
        """
        mesh = fieldkit.Mesh().from_lattice(N=3, lattice=fieldkit.HOOMDLattice(L=3.0))

        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        field[:,:,0] = 0
        domain = fieldkit.domain.digitize(field, threshold=0.5)

        traj,x,im = fieldkit.simulate.random_walk(domain, N=2, steps=1, runs=10)

        # check shape of output is OK
        self.assertEqual(traj.shape, (10,2,3))
        self.assertEqual(x.shape, (2,3))
        self.assertEqual(im.shape, (2,3))

        # check that all coords are still in box
        self.assertTrue(np.all(x >= 0))
        self.assertTrue(np.all(x < 3))

        # walk cannot enter x = 0
        self.assertTrue(np.all(traj[:,:,0] != 0))

        # with 10 steps, a particle cannot have traveled more than 3 images
        self.assertTrue(np.all(im >= -3))
        self.assertTrue(np.all(im < 3))

        # check that trajectory is continuous (no step is larger than 1)
        # 0->1
        self.assertLessEqual(np.max(traj[1]-traj[0]), 1)
        self.assertGreaterEqual(np.min(traj[1]-traj[0]), -1)
        # 1->2
        self.assertLessEqual(np.max(traj[2]-traj[1]), 1)
        self.assertGreaterEqual(np.min(traj[2]-traj[1]), -1)
        # 2->3
        self.assertLessEqual(np.max(traj[3]-traj[2]), 1)
        self.assertGreaterEqual(np.min(traj[3]-traj[2]), -1)

        # try to restart from last state
        traj2,_,_ = fieldkit.simulate.random_walk(domain, N=2, steps=1, runs=1, coords=x, images=im)
        # first frame should match old coordinates
        np.testing.assert_array_equal(traj2[0], x + im*mesh.shape)
        # difference between last old and first new should be 1 step at most
        self.assertLessEqual(np.max(traj2[0]-traj[-1]), 1)
        self.assertGreaterEqual(np.min(traj2[0]-traj[-1]), -1)

    def test_msd(self):
        """ Validate random walk with a short simulation, computing the MSD.

        The simulation is constructed so that the MSD = 1 for each component after 1 run.

        """
        mesh = fieldkit.Mesh().from_lattice(N=10, lattice=fieldkit.HOOMDLattice(L=10.0))
        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        domain = fieldkit.domain.digitize(field, threshold=0.5)

        # displacement over 5 steps should be consistent with random walk
        traj,_,_ = fieldkit.simulate.random_walk(domain, N=1000, steps=3, runs=1000)
        msd = np.zeros(3)
        for i,ri in enumerate(traj[:-1]):
            rj = traj[i+1]
            dr = rj-ri
            msd += np.mean(dr*dr,axis=0)
        msd /= len(traj)-1

        np.testing.assert_array_almost_equal(msd, (1.,1.,1.), decimal=3)
