""" Unit tests for mesh-related data structures.
"""
import unittest
import numpy as np
import fieldkit

class MeshTest(unittest.TestCase):
    """ Test cases for :py:class:`~fieldkit.mesh.Mesh`
    """

    def test_cube(self):
        """ Test mesh formation in simple cube.
        """
        mesh = fieldkit.Mesh().from_lattice(N=4,lattice=fieldkit.HOOMDLattice(L=2.0))
        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.shape, (4,4,4))
        np.testing.assert_array_almost_equal(mesh.step, (0.5, 0.5, 0.5))

    def test_ortho(self):
        """ Test mesh formation in orthorhombic box.
        """
        mesh = fieldkit.Mesh().from_lattice(N=(2,3,4),lattice=fieldkit.HOOMDLattice(L=2.4))
        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.shape, (2,3,4))
        np.testing.assert_array_almost_equal(mesh.step, (1.2, 0.8, 0.6))

        mesh = fieldkit.Mesh().from_lattice(N=(2,3,4),lattice=fieldkit.HOOMDLattice(L=(0.2,0.3,0.4)))
        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.shape, (2,3,4))
        np.testing.assert_array_almost_equal(mesh.step, (0.1, 0.1, 0.1))

    def test_tilt(self):
        """ Test mesh formation with tilt factors.
        """
        mesh = fieldkit.Mesh().from_lattice(N=4,lattice=fieldkit.HOOMDLattice(L=4.,tilt=(0.,0.,0.)))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (3., 3., 3.))

        mesh = fieldkit.Mesh().from_lattice(N=4,lattice=fieldkit.HOOMDLattice(L=4.,tilt=(0.5,0.,0.)))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (4.5, 3., 3.))

        mesh = fieldkit.Mesh().from_lattice(N=4,lattice=fieldkit.HOOMDLattice(L=4.,tilt=(0.5,0.,0.5)))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (4.5, 4.5, 3.))

        mesh = fieldkit.Mesh().from_lattice(N=4,lattice=fieldkit.HOOMDLattice(L=4.,tilt=(0.5,0.5,0.5)))
        np.testing.assert_almost_equal(mesh.grid[-1,-1,-1], (6.0, 4.5, 3.))

    def test_array(self):
        """ Test that mesh can be copied from an existing lattice.
        """
        mesh = fieldkit.Mesh().from_lattice(N=(2,3,4),lattice=fieldkit.HOOMDLattice(L=(0.2,0.3,0.4), tilt=(0.1,0.2,0.3)))
        mesh2 = fieldkit.Mesh().from_array(mesh.grid)

        np.testing.assert_almost_equal(mesh.lattice.L, mesh2.lattice.L)
        np.testing.assert_almost_equal(mesh.lattice.matrix, mesh2.lattice.matrix)

    def test_neighbors(self):
        """ Test for determination of neighbor sites in mesh.
        """
        mesh = fieldkit.Mesh().from_lattice(N=3,lattice=fieldkit.HOOMDLattice(L=1))

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
        mesh = fieldkit.Mesh().from_lattice(N=2,lattice=fieldkit.HOOMDLattice(L=1))
        neigh = mesh.neighbors((0,0,0))
        self.assertEqual(len(neigh), 3)
        np.testing.assert_array_equal(neigh, ((1,0,0),(0,1,0),(0,0,1)))

        neigh = mesh.neighbors((1,1,1))
        self.assertEqual(len(neigh), 3)
        np.testing.assert_array_equal(neigh, ((0,1,1),(1,0,1),(1,1,0)))

        # test stupidly small cells that can't have neighbors
        mesh = fieldkit.Mesh().from_lattice(N=1,lattice=fieldkit.HOOMDLattice(L=1))
        neigh = mesh.neighbors((0,0,0))
        self.assertEqual(len(neigh),0)

    def test_indices(self):
        """ Test for calculation of indices for mesh nodes.
        """
        mesh = fieldkit.Mesh().from_lattice(N=(2,3,4), lattice=fieldkit.HOOMDLattice(L=2.0))
        idx = mesh.indices
        self.assertEqual(idx.shape,(2,3,4,3))
        # reference is given by numpy indexes
        ref = [i for i in np.ndindex(mesh.shape)]
        np.testing.assert_array_equal(idx.reshape((2*3*4,3)), ref)

class FieldTest(unittest.TestCase):
    """ Test cases for :py:class:`~fieldkit.mesh.Field`.
    """
    def setUp(self):
        self.mesh = fieldkit.Mesh().from_lattice(N=(2,3,4),lattice=fieldkit.HOOMDLattice(L=2.0))

    def test(self):
        """ Test for basic creation of an empty field.
        """
        field = fieldkit.Field(self.mesh)
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
        field = fieldkit.Field(self.mesh).from_array(data)
        self.assertEqual(field.shape, self.mesh.shape)
        np.testing.assert_almost_equal(field.field, np.ones(self.mesh.shape))

    def test_multi(self):
        """ Test common multidimensional layouts for input arrays.
        """
        # last index is density
        data = np.ones(np.append(self.mesh.shape, 2))
        field = fieldkit.Field(self.mesh).from_array(data, index=0, axis=3)
        self.assertEqual(field.shape, self.mesh.shape)
        np.testing.assert_almost_equal(field.field, np.ones(self.mesh.shape))

        # first index is density
        data = np.ones(np.insert(self.mesh.shape, 0, 2))
        data[1] *= 2.
        field = fieldkit.Field(self.mesh).from_array(data, index=1, axis=0)
        self.assertEqual(field.shape, self.mesh.shape)
        np.testing.assert_almost_equal(field.field, np.full(self.mesh.shape,2.))

    def test_interpolator(self):
        """ Test for creation of interpolator into field.
        """
        field = fieldkit.Field(self.mesh)
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

class TriangulatedSurfaceTest(unittest.TestCase):
    """ Test cases for :py:class:`~fieldkit.mesh.TriangulatedSurface`
    """

    def test(self):
        """ Test for basic storage of vertices and faces.
        """
        surface = fieldkit.TriangulatedSurface()
        self.assertEqual(surface.vertex.shape, (0,3))
        self.assertEqual(surface.normal.shape, (0,3))
        self.assertEqual(len(surface.face), 0)

        # add one vertex
        surface.add_vertex(vertex=(0,0,0), normal=(-1,0,0))
        self.assertEqual(surface.vertex.shape, (1,3))
        self.assertEqual(surface.normal.shape, (1,3))
        np.testing.assert_almost_equal(surface.vertex, ((0,0,0),) )
        np.testing.assert_almost_equal(surface.normal, ((-1,0,0),) )

        # add two vertices
        surface.add_vertex(vertex=((1,0,0),(0.5,0.5,0)), normal=((1,0,0),(0,1,0)))
        self.assertEqual(surface.vertex.shape, (3,3))
        self.assertEqual(surface.normal.shape, (3,3))
        np.testing.assert_almost_equal(surface.vertex, ((0,0,0),(1,0,0),(0.5,0.5,0)) )
        np.testing.assert_almost_equal(surface.normal, ((-1,0,0),(1,0,0),(0,1,0)) )

        # add a face
        surface.add_face((0,1,2))
        self.assertEqual(len(surface.face),1)
        self.assertEqual(surface.face, ((0,1,2),))
        surface.add_face(((1,2,0),(2,0,1)))
        self.assertEqual(len(surface.face),3)
        self.assertEqual(surface.face, ((0,1,2),(1,2,0),(2,0,1)))

    def test_exceptions(self):
        """ Test for exception throwing adding vertices and faces.
        """
        surface = fieldkit.TriangulatedSurface()

        # check vector sizes
        with self.assertRaises(IndexError):
            surface.add_vertex(vertex=(0,0), normal=(1,0,0))
        with self.assertRaises(IndexError):
            surface.add_vertex(vertex=(0,0,0), normal=(1,0))
        with self.assertRaises(IndexError):
            surface.add_vertex(vertex=((0,0),(0,0)), normal=((1,0,0),(1,0,0)))
        with self.assertRaises(IndexError):
            surface.add_vertex(vertex=((0,0,0),(0,0,0)), normal=((1,0),(1,0)))

        # check mismatch in numbers
        with self.assertRaises(IndexError):
            surface.add_vertex(vertex=((0,0,0),(0,0,0)), normal=((1,0,0),))
        with self.assertRaises(IndexError):
            surface.add_vertex(vertex=((0,0,0),), normal=((1,0,0),(1,0,0)))

        # check number of vertices in faces
        with self.assertRaises(IndexError):
            surface.add_face((0,1))
        with self.assertRaises(IndexError):
            surface.add_face((0,1,2,3))

class DomainTest(unittest.TestCase):
    """ Test cases for :py:class:`~fieldkit.mesh.Domain`
    """
    def setUp(self):
        self.mesh = fieldkit.Mesh().from_lattice(N=(3,4,5),lattice=fieldkit.HOOMDLattice(L=(1.5,2,2.5)))

    def test(self):
        """ Basic tests of constructing domain and its graph.
        """
        nodes = self.mesh.indices.reshape((np.prod(self.mesh.shape), 3))
        domain = fieldkit.Domain(self.mesh, nodes)
        self.assertEqual(domain.mesh, self.mesh)
        np.testing.assert_array_equal(domain.nodes, nodes)

        # check for mask being made and cached
        self.assertTrue(domain._mask is None)
        np.testing.assert_array_equal(self.mesh.indices[domain.mask], nodes)
        self.assertTrue(domain._mask is not None)

        # graph is initially not built
        self.assertTrue(domain._graph is None)

        # base graph is all nodes connected together
        graph = domain.graph
        self.assertEqual(len(graph.nodes()), 3*4*5)
        edges = graph.edges()
        self.assertEqual(len(edges), 3*4*5*3)
        # x periodic
        self.assertTrue(((0,0,0),(2,0,0)) in edges)
        self.assertTrue(((0,3,4),(2,3,4)) in edges)
        # y periodic
        self.assertTrue(((0,0,0),(0,3,0)) in edges)
        self.assertTrue(((2,0,4),(2,3,4)) in edges)
        # z periodic
        self.assertTrue(((0,0,0),(0,0,4)) in edges)
        self.assertTrue(((2,3,0),(2,3,4)) in edges)
        # check weights of edges, which should all be 0.5
        weights = [e[2] for e in graph.edges(data='weight')]
        np.testing.assert_almost_equal(weights, 0.5*np.ones(3*4*5*3))

        # check graph is cached
        self.assertTrue(domain._graph is not None)

    def test_buffered_graph(self):
        """ Tests for constructing buffered graph along an axis.
        """
        nodes = self.mesh.indices.reshape((np.prod(self.mesh.shape), 3))
        domain = fieldkit.Domain(self.mesh, nodes)

        # test for basic buffering behavior in x
        buff = domain.buffered_graph(axis=0)
        # number of nodes increases by 1 layer
        self.assertEqual(len(buff.nodes()), 4*4*5)
        # the number of edges is the same as unbuffered
        self.assertEqual(len(buff.edges()), 3*4*5*3)
        edges = buff.edges()
        # x buffered
        self.assertTrue(((0,0,0),(2,0,0)) not in edges)
        self.assertTrue(((2,0,0),(3,0,0)) in edges)
        self.assertTrue(((0,3,4),(2,3,4)) not in edges)
        self.assertTrue(((2,3,4),(3,3,4)) in edges)
        # y periodic
        self.assertTrue(((0,0,0),(0,3,0)) in edges)
        self.assertTrue(((2,0,4),(2,3,4)) in edges)
        # z periodic
        self.assertTrue(((0,0,0),(0,0,4)) in edges)
        self.assertTrue(((2,3,0),(2,3,4)) in edges)
        # check weights of edges, which should all be 0.5
        weights = [e[2] for e in buff.edges(data='weight')]
        np.testing.assert_almost_equal(weights, 0.5*np.ones(3*4*5*3))

        # test for basic buffering behavior in y
        buff = domain.buffered_graph(axis=1)
        self.assertEqual(len(buff.nodes()), 3*5*5)
        self.assertEqual(len(buff.edges()), 3*4*5*3)
        edges = buff.edges()
        # x periodic
        self.assertTrue(((0,0,0),(2,0,0)) in edges)
        self.assertTrue(((0,3,4),(2,3,4)) in edges)
        # y buffered
        self.assertTrue(((0,0,0),(0,3,0)) not in edges)
        self.assertTrue(((0,3,0),(0,4,0)) in edges)
        self.assertTrue(((2,0,4),(2,3,4)) not in edges)
        self.assertTrue(((2,3,4),(2,4,4)) in edges)
        # z periodic
        self.assertTrue(((0,0,0),(0,0,4)) in edges)
        self.assertTrue(((2,3,0),(2,3,4)) in edges)
        # check weights of edges, which should all be 0.5
        weights = [e[2] for e in buff.edges(data='weight')]
        np.testing.assert_almost_equal(weights, 0.5*np.ones(3*4*5*3))

        # test for basic buffering behavior in z
        buff = domain.buffered_graph(axis=2)
        self.assertEqual(len(buff.nodes()), 3*4*6)
        self.assertEqual(len(buff.edges()), 3*4*5*3)
        edges = buff.edges()
        # x periodic
        self.assertTrue(((0,0,0),(2,0,0)) in edges)
        self.assertTrue(((0,3,4),(2,3,4)) in edges)
        # y periodic
        self.assertTrue(((0,0,0),(0,3,0)) in edges)
        self.assertTrue(((2,0,4),(2,3,4)) in edges)
        # z buffered
        self.assertTrue(((0,0,0),(0,0,4)) not in edges)
        self.assertTrue(((0,0,4),(0,0,5)) in edges)
        self.assertTrue(((2,3,0),(2,3,4)) not in edges)
        self.assertTrue(((2,3,4),(2,3,5)) in edges)
        # check weights of edges, which should all be 0.5
        weights = [e[2] for e in buff.edges(data='weight')]
        np.testing.assert_almost_equal(weights, 0.5*np.ones(3*4*5*3))

    def test_slab(self):
        """ Application test for a slab domain.
        """
        nodes = [(i,j,1) for i in range(self.mesh.shape[0]) for j in range(self.mesh.shape[1])]
        domain = fieldkit.Domain(self.mesh, nodes)

        # graph is only 2d periodic (in x and y)
        graph = domain.graph
        self.assertEqual(len(graph.nodes()), 3*4)
        self.assertEqual(len(graph.edges()), 3*4*2)

        # buffering in z should do nothing since domain is not spanning
        buff = domain.buffered_graph(axis=2)
        self.assertEqual(len(buff.nodes()), 3*4)
        self.assertEqual(len(buff.edges()), 3*4*2)

        # buffering x should expand by 1 node
        buff = domain.buffered_graph(axis=0)
        self.assertEqual(len(buff.nodes()), 4*4)
        self.assertEqual(len(buff.edges()), 3*4*2)

        # buffering y should expand by 1 node
        buff = domain.buffered_graph(axis=1)
        self.assertEqual(len(buff.nodes()), 3*5)
        self.assertEqual(len(buff.edges()), 3*4*2)
