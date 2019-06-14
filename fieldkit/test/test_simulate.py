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

        # walk cannot enter z = 0
        self.assertTrue(np.all(traj[:,:,2] != 0))

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

        # displacement should be consistent with random walk
        traj,_,_ = fieldkit.simulate.random_walk(domain, N=4000, steps=3, runs=1000)
        window = 3
        msd = np.zeros((window+1,3))
        samples = np.zeros(window+1, dtype=np.int32)
        for i,ri in enumerate(traj[:-1]):
            for dt in range(1,min(window+1,traj.shape[0]-i)):
                rj = traj[i+dt]
                dr = rj-ri
                msd[dt] += np.mean(dr*dr,axis=0)
                samples[dt] += 1
        flags = samples > 0
        for ax in range(3):
            msd[flags,ax] /= samples[flags]
        np.testing.assert_array_almost_equal(msd[0], (0.,0.,0.), decimal=3)
        np.testing.assert_array_almost_equal(msd[1], (1.,1.,1.), decimal=3)
        np.testing.assert_array_almost_equal(msd[2], (2.,2.,2.), decimal=2)
        np.testing.assert_array_almost_equal(msd[3], (3.,3.,3.), decimal=2)

        # use compiled code to test farther out
        msd_2 = fieldkit.simulate.msd(traj,window=window)
        self.assertEqual(msd_2.shape, (window+1,3))
        np.testing.assert_array_almost_equal(msd_2[0], (0.,0.,0.), decimal=3)
        np.testing.assert_array_almost_equal(msd_2[1], (1.,1.,1.), decimal=3)
        np.testing.assert_array_almost_equal(msd_2[2], (2.,2.,2.), decimal=2)
        np.testing.assert_array_almost_equal(msd_2[3], (3.,3.,3.), decimal=2)

        # both results should be essentially the same
        np.testing.assert_array_almost_equal(msd,msd_2)

        # use every 2nd origin with a looser tolerance due to lower stats
        msd_3 = fieldkit.simulate.msd(traj,window=window,every=2)
        self.assertEqual(msd_3.shape, (window+1,3))
        np.testing.assert_array_almost_equal(msd_3[0], (0.,0.,0.), decimal=3)
        np.testing.assert_array_almost_equal(msd_3[1], (1.,1.,1.), decimal=2)
        np.testing.assert_array_almost_equal(msd_3[2], (2.,2.,2.), decimal=2)
        np.testing.assert_array_almost_equal(msd_3[3], (3.,3.,3.), decimal=2)

    def test_msd_binned(self):
        """ Test binned MSD compared to bulk MSD calculator.

        The simulation is constructed so that the MSD = 1 for each component after 1 run.

        """
        # dummy trajectory
        traj = np.zeros((4,3,3))
        traj[0,:] = [[0,0,0],[-1.9, 0, 0],[1.5,3,7]]
        traj[1,:] = [[0.1,2,-1],[-1.8,-1,3],[1.6,4,8]]
        traj[2,:] = [[0.2,4,-2],[-1.7,-2,6],[1.7,5,9]]
        traj[3,:] = [[0.3,6,-3],[-1.6,-3,9],[1.8,6,10]]

        # msd from binned
        msd_bin,edges = fieldkit.simulate.msd_binned(traj, window=1, axis=0, bins=8, range=(-2,2))
        self.assertEqual(msd_bin.shape, (8,2,3))
        self.assertEqual(edges.shape, (9,))
        np.testing.assert_array_almost_equal(edges,(-2.,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0))

        # only bins 0, 4, and 7 have particles contributing
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.,0.),(1.e-2,1.,9.)))
        np.testing.assert_array_almost_equal(msd_bin[1], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[2], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[3], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[4], ((0.,0.,0.),(1.e-2,4.,1.)))
        np.testing.assert_array_almost_equal(msd_bin[5], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[6], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[7], ((0.,0.,0.),(1.e-2,1.,1.)))

        # repeat using every other origin, should give identical result
        msd_bin,_ = fieldkit.simulate.msd_binned(traj, window=1, axis=0, bins=8, range=(-2,2), every=2)
        self.assertEqual(msd_bin.shape, (8,2,3))
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.,0.),(1.e-2,1.,9.)))
        np.testing.assert_array_almost_equal(msd_bin[1], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[2], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[3], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[4], ((0.,0.,0.),(1.e-2,4.,1.)))
        np.testing.assert_array_almost_equal(msd_bin[5], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[6], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[7], ((0.,0.,0.),(1.e-2,1.,1.)))

        # compute with a range that no particles lie in, should give all zeros
        msd_bin,_ = fieldkit.simulate.msd_binned(traj, window=1, axis=0, bins=1, range=(-1.5,-0.1))
        self.assertEqual(msd_bin.shape, (1,2,3))
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.,0.),(0.,0.,0.)))

        # repeat for the window that only the first particle lies in
        msd_bin,_ = fieldkit.simulate.msd_binned(traj, window=1, axis=0, bins=3, range=(0,0.6))
        self.assertEqual(msd_bin.shape, (3,2,3))
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.,0.),(1.e-2,4.,1.)))
        np.testing.assert_array_almost_equal(msd_bin[1], ((0.,0.,0.),(1.e-2,4.,1.)))
        np.testing.assert_array_almost_equal(msd_bin[2], ((0.,0.,0.),(0.,0.,0.)))

        # roll the trajectory so binning is done along y
        traj = np.roll(traj, shift=1, axis=2)
        msd_bin,_ = fieldkit.simulate.msd_binned(traj, window=1, axis=1, bins=8, range=(-2,2))
        self.assertEqual(msd_bin.shape, (8,2,3))
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.,0.),(9.,1.e-2,1.)))
        np.testing.assert_array_almost_equal(msd_bin[1], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[2], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[3], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[4], ((0.,0.,0.),(1.,1.e-2,4.)))
        np.testing.assert_array_almost_equal(msd_bin[5], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[6], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[7], ((0.,0.,0.),(1.,1.e-2,1.)))

        # roll again so binning is done along z
        traj = np.roll(traj, shift=1, axis=2)
        msd_bin,_ = fieldkit.simulate.msd_binned(traj, window=1, axis=2, bins=8, range=(-2,2))
        self.assertEqual(msd_bin.shape, (8,2,3))
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.,0.),(1.,9.,1.e-2)))
        np.testing.assert_array_almost_equal(msd_bin[1], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[2], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[3], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[4], ((0.,0.,0.),(4.,1.,1.e-2)))
        np.testing.assert_array_almost_equal(msd_bin[5], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[6], ((0.,0.,0.),(0.,0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[7], ((0.,0.,0.),(1.,1.,1.e-2)))
