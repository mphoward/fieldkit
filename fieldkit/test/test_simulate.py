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

        traj,x,im = fieldkit.simulate.random_walk(domain, N=2, steps=1, runs=10, seed=42)

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
        traj2,_,_ = fieldkit.simulate.random_walk(domain, N=2, steps=1, runs=1, coords=x, images=im, seed=24)
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
        traj,_,_ = fieldkit.simulate.random_walk(domain, N=4000, steps=3, runs=1000, seed=42)
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

    def test_msd_survival(self):
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
        msd_bin,counts,edges = fieldkit.simulate.msd_survival(traj, window=1, axis=0, bins=8, range=(-2,2))
        self.assertEqual(msd_bin.shape, (8,2,2))
        self.assertEqual(counts.shape, (8,2))
        self.assertEqual(edges.shape, (9,))
        np.testing.assert_array_almost_equal(edges,(-2.,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0))

        # check counts
        np.testing.assert_array_equal(counts[0], (3,3))
        np.testing.assert_array_equal(counts[1], (0,0))
        np.testing.assert_array_equal(counts[2], (0,0))
        np.testing.assert_array_equal(counts[3], (0,0))
        np.testing.assert_array_equal(counts[4], (3,3))
        np.testing.assert_array_equal(counts[5], (0,0))
        np.testing.assert_array_equal(counts[6], (0,0))
        np.testing.assert_array_equal(counts[7], (3,3))

        # only bins 0, 4, and 7 have particles contributing
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.),(1.,9.)))
        np.testing.assert_array_almost_equal(msd_bin[1], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[2], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[3], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[4], ((0.,0.),(4.,1.)))
        np.testing.assert_array_almost_equal(msd_bin[5], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[6], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[7], ((0.,0.),(1.,1.)))

        # repeat using every other origin, should give identical result
        msd_bin,_,_ = fieldkit.simulate.msd_survival(traj, window=1, axis=0, bins=8, range=(-2,2), every=2)
        self.assertEqual(msd_bin.shape, (8,2,2))
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.),(1.,9.)))
        np.testing.assert_array_almost_equal(msd_bin[1], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[2], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[3], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[4], ((0.,0.),(4.,1.)))
        np.testing.assert_array_almost_equal(msd_bin[5], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[6], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[7], ((0.,0.),(1.,1.)))

        # compute with a range that no particles lie in, should give all zeros
        msd_bin,_,_ = fieldkit.simulate.msd_survival(traj, window=1, axis=0, bins=1, range=(-1.5,-0.1))
        self.assertEqual(msd_bin.shape, (1,2,2))
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.),(0.,0.)))

        # roll the trajectory so binning is done along y
        traj = np.roll(traj, shift=1, axis=2)
        msd_bin,_,_ = fieldkit.simulate.msd_survival(traj, window=1, axis=1, bins=8, range=(-2,2))
        self.assertEqual(msd_bin.shape, (8,2,2))
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.),(9.,1.)))
        np.testing.assert_array_almost_equal(msd_bin[1], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[2], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[3], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[4], ((0.,0.),(1.,4.)))
        np.testing.assert_array_almost_equal(msd_bin[5], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[6], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[7], ((0.,0.),(1.,1.)))

        # roll again so binning is done along z
        traj = np.roll(traj, shift=1, axis=2)
        msd_bin,_,_ = fieldkit.simulate.msd_survival(traj, window=1, axis=2, bins=8, range=(-2,2))
        self.assertEqual(msd_bin.shape, (8,2,2))
        np.testing.assert_array_almost_equal(msd_bin[0], ((0.,0.),(1.,9.)))
        np.testing.assert_array_almost_equal(msd_bin[1], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[2], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[3], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[4], ((0.,0.),(4.,1.)))
        np.testing.assert_array_almost_equal(msd_bin[5], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[6], ((0.,0.),(0.,0.)))
        np.testing.assert_array_almost_equal(msd_bin[7], ((0.,0.),(1.,1.)))

        # TODO: test when a particle exits the bin

    def test_msd_survival_cylinder(self):
        """ Test radially binned MSD."""
        # dummy radial and axial coordinates
        r = np.zeros((4,3))
        r[0,:] = [0.0,1.5,2.5]
        r[1,:] = [0.3,1.3,2.5]
        r[2,:] = [0.6,1.1,2.5]
        r[3,:] = [0.9,1.01,2.5]
        # 0 has D = 1, 1 has D = 2, 2 has D = 3
        z = np.zeros((4,3))
        z[0,:] = [1,-2,0]
        z[1,:] = [2,-4,3]
        z[2,:] = [3,-6,6]
        z[3,:] = [4,-8,9]

        # msd from binned
        msd_bin,counts,edges = fieldkit.simulate.msd_survival_cylinder(r, z, window=1, bins=4, range=(0,4))
        self.assertEqual(msd_bin.shape, (4,2))
        self.assertEqual(counts.shape, (4,2))
        self.assertEqual(edges.shape, (5,))
        np.testing.assert_array_almost_equal(edges,(0,1,2,3,4))

        # check counts
        np.testing.assert_array_equal(counts[0], (3,3))
        np.testing.assert_array_equal(counts[1], (3,3))
        np.testing.assert_array_equal(counts[2], (3,3))
        np.testing.assert_array_equal(counts[3], (0,0))

        # only bins 0, 1, and 2 have particles contributing
        np.testing.assert_array_almost_equal(msd_bin[0], (0,1))
        np.testing.assert_array_almost_equal(msd_bin[1], (0,4))
        np.testing.assert_array_almost_equal(msd_bin[2], (0,9))
        np.testing.assert_array_almost_equal(msd_bin[3], (0,0))

        # repeat using every other origin, should give identical result
        msd_bin,_,_ = fieldkit.simulate.msd_survival_cylinder(r, z, window=1, bins=4, range=(0,4), every=2)
        self.assertEqual(msd_bin.shape, (4,2))
        np.testing.assert_array_almost_equal(msd_bin[0], (0,1))
        np.testing.assert_array_almost_equal(msd_bin[1], (0,4))
        np.testing.assert_array_almost_equal(msd_bin[2], (0,9))
        np.testing.assert_array_almost_equal(msd_bin[3], (0,0))

        # shring range to lose inner and outer particle
        msd_bin,_,_ = fieldkit.simulate.msd_survival_cylinder(r, z, window=1, bins=2, range=(1,3))
        self.assertEqual(msd_bin.shape, (2,2))
        np.testing.assert_array_almost_equal(msd_bin[0], (0,4))
        np.testing.assert_array_almost_equal(msd_bin[1], (0,9))

class BiasedRandomWalkTest(unittest.TestCase):
    """ Test cases for :py:class:`~fieldkit.simulate.biased_walk`
    """

    def test_one_step(self):
        """ Test biased random walk rules for one step, using unbiased rates.
        """
        mesh = fieldkit.Mesh().from_lattice(N=3, lattice=fieldkit.HOOMDLattice(L=3.0))

        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        field[:,:,0] = 0
        domain = fieldkit.domain.digitize(field, threshold=0.5)

        # these are the hopping rates.
        # should really be zero to go in boundary, but this move will be rejected anyway.
        probs = np.full(list(mesh.shape) + [6], 1./6.)

        traj,x,im = fieldkit.simulate.biased_walk(domain, probs, N=2, steps=1, runs=10, seed=42)

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
        traj2,_,_ = fieldkit.simulate.biased_walk(domain, probs, N=2, steps=1, runs=1, coords=x, images=im, seed=24)
        # first frame should match old coordinates
        np.testing.assert_array_equal(traj2[0], x + im*mesh.shape)
        # difference between last old and first new should be 1 step at most
        self.assertLessEqual(np.max(traj2[0]-traj[-1]), 1)
        self.assertGreaterEqual(np.min(traj2[0]-traj[-1]), -1)

    def test_msd(self):
        """ Validate biased random walk with a short simulation, computing the MSD.

        The simulation is constructed so that the MSD = 1 for each component after 1 run.

        """
        mesh = fieldkit.Mesh().from_lattice(N=10, lattice=fieldkit.HOOMDLattice(L=10.0))
        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        domain = fieldkit.domain.digitize(field, threshold=0.5)

        # these are the hopping rates, which we make a random walk for now
        probs = np.full(list(mesh.shape) + [6], 1./6.)

        # displacement should be consistent with random walk
        traj,_,_ = fieldkit.simulate.biased_walk(domain, probs, N=4000, steps=3, runs=1000, seed=42)
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

class kmcTest(unittest.TestCase):
    """ Test cases for :py:class:`~fieldkit.simulate.kmc`"""

    def test_basic(self):
        """ Test basic biased random walk rules, using unbiased rates."""
        mesh = fieldkit.Mesh().from_lattice(N=3, lattice=fieldkit.HOOMDLattice(L=3.0))

        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        field[:,:,0] = 0
        domain = fieldkit.domain.digitize(field, threshold=0.5)

        # these are the hopping rates.
        # should really be zero to go in boundary, but this move will be rejected anyway.
        rates = np.full(list(mesh.shape) + [6], 1.)
        traj,x,im,t = fieldkit.simulate.kmc(domain, rates, np.arange(10), N=2, steps=100, seed=42)

        # check shape of output is OK
        self.assertEqual(traj.shape, (10,2,3))
        self.assertEqual(x.shape, (2,3))
        self.assertEqual(im.shape, (2,3))
        self.assertEqual(t.shape, (2,))

        # check that all coords are still in box
        self.assertTrue(np.all(x >= 0))
        self.assertTrue(np.all(x < 3))

        # walk cannot enter z = 0
        self.assertTrue(np.all(traj[:,:,2] != 0))

        # with 10 steps, a particle cannot have traveled more than 3 images
        self.assertTrue(np.all(im >= -3))
        self.assertTrue(np.all(im < 3))

    def test_msd(self):
        """ Validate biased random walk with a short simulation, computing the MSD.

        This MSD is a little different than the usual random walk because the MSD tends
        to 1.0 at t->0, rather than 0.0, because any hop (even after a short time) moves
        the walker by one lattice site.

        """
        mesh = fieldkit.Mesh().from_lattice(N=10, lattice=fieldkit.HOOMDLattice(L=10.0))
        field = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        domain = fieldkit.domain.digitize(field, threshold=0.5)

        # displacement should be consistent with random walk with coeff. 1/2
        rates = np.full(list(mesh.shape) + [6], 0.5)
        traj,_,_,_ = fieldkit.simulate.kmc(domain, rates, np.arange(2000), N=4000, steps=10000, seed=42)
        window = 3
        msd = fieldkit.simulate.msd(traj,window=window)
        self.assertEqual(msd.shape, (window+1,3))
        np.testing.assert_array_almost_equal(msd[0], (0.,0.,0.), decimal=3)
        np.testing.assert_array_almost_equal(msd[1], (1.,1.,1.), decimal=3)
        np.testing.assert_array_almost_equal(msd[2], (2.,2.,2.), decimal=2)
        np.testing.assert_array_almost_equal(msd[3], (3.,3.,3.), decimal=2)

    def test_rates(self):
        """Test hopping rate calculator."""
        mesh = fieldkit.Mesh().from_lattice(N=12, lattice=fieldkit.HOOMDLattice(L=12.0))

        ## bias D in z
        flags = np.logical_or(mesh.grid[...,2] < 1.0, mesh.grid[...,2] > 10)
        D = fieldkit.Field(mesh).from_array(0.1*(11.0-mesh.grid[...,2]))
        D[flags] = 0.
        rho = fieldkit.Field(mesh).from_array(np.ones(mesh.shape))
        rho[flags] = 0.
        domain = fieldkit.domain.digitize(rho, threshold=1.e-6)

        # check rates by hand for simple case
        rates = fieldkit.simulate.compute_hopping_rates(domain, D, rho)
        self.assertEqual(rates.shape, (12,12,12,6))
        np.testing.assert_array_almost_equal(rates[0,0,0], (0.,0.,0.,0.,0.,0.))
        np.testing.assert_array_almost_equal(rates[0,0,1], (1.,1.,1.,1.,0.95,0.))
        np.testing.assert_array_almost_equal(rates[0,0,2], (0.9,0.9,0.9,0.9,0.85,0.95))
        # ...
        np.testing.assert_array_almost_equal(rates[0,0,9], (0.2,0.2,0.2,0.2,0.15,0.25))
        np.testing.assert_array_almost_equal(rates[0,0,10], (0.1,0.1,0.1,0.1,0.,0.15))
        np.testing.assert_array_almost_equal(rates[0,0,11], (0.,0.,0.,0.,0.,0.))
        np.testing.assert_array_almost_equal(rates[0,0], rates[1,0])
        np.testing.assert_array_almost_equal(rates[0,0], rates[0,1])

        # run short simulation to verify density distribution
        traj,_,_,_ = fieldkit.simulate.kmc(domain, rates, 100+np.arange(500), N=1000, steps=10000, seed=42)
        self.assertEqual(np.min(traj[...,2]), 1)
        self.assertEqual(np.max(traj[...,2]), 10)
        hist,_ = np.histogram(traj[...,2], range=(0.5,10.5), bins=10, density=True)
        np.testing.assert_array_almost_equal(hist, 0.1, decimal=2)

        ## also bias the density in z
        rho.field = 0.1*(mesh.grid[...,2])
        rho[flags] = 0.
        domain = fieldkit.domain.digitize(rho, threshold=1.e-6)
        rates = fieldkit.simulate.compute_hopping_rates(domain, D, rho)
        np.testing.assert_array_almost_equal(rates[0,0,0,0:4], 0.)
        np.testing.assert_array_almost_equal(rates[0,0,1,0:4], 1.)
        np.testing.assert_array_almost_equal(rates[0,0,10,0:4], 0.1)
        np.testing.assert_array_almost_equal(rates[0,0,11,0:4], 0.)
        np.testing.assert_array_almost_equal(rates[0,0], rates[1,0])
        np.testing.assert_array_almost_equal(rates[0,0], rates[0,1])

        traj,_,_,_ = fieldkit.simulate.kmc(domain, rates, 100+np.arange(500), N=1000, steps=10000, seed=42)
        self.assertEqual(np.min(traj[...,2]), 1)
        self.assertEqual(np.max(traj[...,2]), 10)
        hist,_ = np.histogram(traj[...,2], range=(0.5,10.5), bins=10, density=True)
        np.testing.assert_array_almost_equal(hist, rho.field[0,0,1:-1]/np.sum(rho.field[0,0,1:-1]), decimal=2)

        ## roll axis to y
        D.field = np.moveaxis(D.field,2,1)
        rho.field = np.moveaxis(rho.field,2,1)
        domain = fieldkit.domain.digitize(rho, threshold=1.e-6)
        rates = fieldkit.simulate.compute_hopping_rates(domain, D, rho)
        np.testing.assert_array_almost_equal(rates[0,0,0,0:2], 0.)
        np.testing.assert_array_almost_equal(rates[0,0,0,4:6], 0.)
        np.testing.assert_array_almost_equal(rates[0,1,0,0:2], 1.)
        np.testing.assert_array_almost_equal(rates[0,1,0,4:6], 1.)
        np.testing.assert_array_almost_equal(rates[0,10,0,0:2], 0.1)
        np.testing.assert_array_almost_equal(rates[0,10,0,4:6], 0.1)
        np.testing.assert_array_almost_equal(rates[0,11,0,0:2], 0.)
        np.testing.assert_array_almost_equal(rates[0,11,0,4:6], 0.)
        np.testing.assert_array_almost_equal(rates[0,:,0], rates[1,:,0])
        np.testing.assert_array_almost_equal(rates[0,:,0], rates[0,:,1])

        traj,_,_,_ = fieldkit.simulate.kmc(domain, rates, 100+np.arange(500), N=1000, steps=10000, seed=42)
        self.assertEqual(np.min(traj[...,1]), 1)
        self.assertEqual(np.max(traj[...,1]), 10)
        hist,_ = np.histogram(traj[...,1], range=(0.5,10.5), bins=10, density=True)
        np.testing.assert_array_almost_equal(hist, rho.field[0,1:-1,0]/np.sum(rho.field[0,1:-1,0]), decimal=2)

        ## roll axis to z
        D.field = np.moveaxis(D.field,1,0)
        rho.field = np.moveaxis(rho.field,1,0)
        domain = fieldkit.domain.digitize(rho, threshold=1.e-6)
        rates = fieldkit.simulate.compute_hopping_rates(domain, D, rho)
        np.testing.assert_array_almost_equal(rates[0,0,0,2:6], 0.)
        np.testing.assert_array_almost_equal(rates[1,0,0,2:6], 1.)
        np.testing.assert_array_almost_equal(rates[10,0,0,2:6], 0.1)
        np.testing.assert_array_almost_equal(rates[11,0,0,2:6], 0.)
        np.testing.assert_array_almost_equal(rates[:,0,0], rates[:,1,0])
        np.testing.assert_array_almost_equal(rates[:,0,0], rates[:,0,1])

        traj,_,_,_ = fieldkit.simulate.kmc(domain, rates, 100+np.arange(500), N=1000, steps=10000, seed=42)
        self.assertEqual(np.min(traj[...,0]), 1)
        self.assertEqual(np.max(traj[...,0]), 10)
        hist,_ = np.histogram(traj[...,0], range=(0.5,10.5), bins=10, density=True)
        np.testing.assert_array_almost_equal(hist, rho.field[1:-1,0,0]/np.sum(rho.field[1:-1,0,0]), decimal=2)
