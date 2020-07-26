import copy
import numpy as np
from ..numerics.fem import domain2d_fem as FEM
from ._mesh import Mesh
from meshpy import triangle
from scipy import sparse
from numba import jit
from numpy.linalg import det


class Domain2d(Mesh):

    def __init__(self, curves=None, triangles=None, vertices=None):
        super(Domain2d, self).__init__()

        if curves is not None:
            triangles, vertices = self.mesh( curves )

        elif (triangles is None) and (vertices is None):
            raise ValueError("Either curves or (triangles,vertices) should be defined!")

        self._triangles = triangles
        self._vertices = vertices
        self.FEM = FEM
        self.ref_element = FEM.ReferenceElement()
        local_to_global = self.ref_element.local_to_global
        self.local_to_global = lambda s, mask, coarsened: \
                               local_to_global(s, self, mask, coarsened)
        self.reset_data()

    def copy(self):
        new_domain =  Domain2d( triangles = self._triangles.copy(),
                                vertices = self._vertices.copy() )

        if self._coords is not None:
            new_domain._coords = copy.deepcopy( self._coords )
        if self._el_sizes is not None:
            new_domain._el_sizes = self._el_sizes.copy()
        if self._area is not None:
            new_domain._area = self._area

        new_domain.timestamp = self.timestamp

        return new_domain


    def dim(self):
        return 2

    def dim_of_world(self):
        return 2

    def reset_data(self):
        self._coords = None
        self._el_sizes = None
        self._area = None
        self.timestamp += 1

    def size(self):
        return self._triangles.shape[1]

    def vertices(self):
        return self._vertices

    def triangles(self):
        return (self._triangles, self._vertices)

    def coords(self,mask=None):
        if mask is None:
            if self._coords is not None:
                return self._coords

        if mask is not None:
            x0 = self._vertices[ :, self._triangles[0,mask] ]
            x1 = self._vertices[ :, self._triangles[1,mask] ]
            x2 = self._vertices[ :, self._triangles[2,mask] ]
            return (x0, x1, x2)

        else: # mask is None, compute coordinate triplets for all elements.

            ## # The old Numpy code for coordinate computation

            ## x0 = self._vertices[ :, self._triangles[0,:] ]
            ## x1 = self._vertices[ :, self._triangles[1,:] ]
            ## x2 = self._vertices[ :, self._triangles[2,:] ]
            ## self._coords = (x0, x1, x2)

            # The following code moved a Numba jit-compiled function
            # is supposed to speed up the coordinate computation.
            # The tests indicate a speed-up of 3X-10X.
            tri = self._triangles
            vtx = self._vertices

            n_tri = tri.shape[1]
            x0 = np.empty( (2,n_tri), dtype=float )
            x1 = np.empty( (2,n_tri), dtype=float )
            x2 = np.empty( (2,n_tri), dtype=float )

            Domain2d._retrieve_coords( vtx, tri, x0,x1,x2 )

            self._coords = (x0, x1, x2)
            return self._coords

    @jit( nopython = True )
    def _retrieve_coords(vtx, tri, x0, x1, x2):
        n_tri = tri.shape[1]
        for i in range(0, n_tri):
            x0[0,i] = vtx[ 0, tri[0,i] ]
            x1[0,i] = vtx[ 0, tri[1,i] ]
            x2[0,i] = vtx[ 0, tri[2,i] ]
            x0[1,i] = vtx[ 1, tri[0,i] ]
            x1[1,i] = vtx[ 1, tri[1,i] ]
            x2[1,i] = vtx[ 1, tri[2,i] ]


    def element_sizes(self,mask=None):
        if mask is None:
            if self._el_sizes is not None:
                return self._el_sizes

        x0, x1, x2 = self.coords()

        if mask is not None:
            d = 0.5 * ( x0[0,mask] * (x1[1,mask] - x2[1,mask]) + \
                        x1[0,mask] * (x2[1,mask] - x0[1,mask]) + \
                        x2[0,mask] * (x0[1,mask] - x1[1,mask]) )
            return d

        else: # mask not given, compute element sizes for all elements.

            ## d = 0.5 * ( x0[0,:] * (x1[1,:] - x2[1,:]) + \
            ##             x1[0,:] * (x2[1,:] - x0[1,:]) + \
            ##             x2[0,:] * (x0[1,:] - x1[1,:]) )

            self._el_sizes = d = Domain2d._fast_element_sizes( x0, x1, x2 )
            return d.copy()

    @jit( nopython = True )
    def _fast_element_sizes(x0, x1, x2):
            n = x0.shape[1]
            d = np.empty(n)
            for i in range(0,n):
                d[i] = 0.5 * ( x0[0,i] * (x1[1,i] - x2[1,i]) +
                               x1[0,i] * (x2[1,i] - x0[1,i]) +
                               x2[0,i] * (x0[1,i] - x1[1,i]) )
            return d

    def area(self):
        if self._area is not None:
            return self._area
        else:
            self._area = np.sum( self.element_sizes() )
            return self._area

    def volume(self):
        return self.area()

    def mesh(self, curves, options='p'):
        if curves.size() < 3:
            raise ValueError("Curve should have at least three vertices.")

        points = curves.coords().T
        segments = curves.edges().T

        info = triangle.MeshInfo()
        info.set_points( points )
        info.set_facets( segments )

        try:
            hole_pts = curves.hole_points().T
            if len(hole_pts) > 0:  info.set_holes( hole_pts )
        except AttributeError: # if input is a single curve, it doesn't have holes() func
            pass

        triangulation = triangle.build(info)

        tri = np.array( triangulation.elements ).T
        vtx = np.array( triangulation.points ).T

        return (tri,vtx)


    ## def _compute_geom_info(self):

    ##     elem = self._triangles
    ##     n = self._vertices.shape[1]
    ##     # Indices of (i,j), and possibly some (j,i), which are the same edges.
    ##     I = elem.flatten()
    ##     J = np.hstack( (elem[1,:], elem[2,:], elem[0,:]) )
    ##     # I2,J2 has all (i,j) and (j,i) converted to (i,j), (i,j).
    ##     # Now we need to get rid of duplicates.
    ##     I2 = np.minimum(I,J)
    ##     J2 = np.maximum(I,J)

    ##     # Use new_edge_id_mtx to get rid of duplicates.
    ##     marker = np.ones( len(I2), dtype=int )
    ##     new_edge_id_mtx = sparse.csr_matrix( (marker, (I2,J2)), (n,n) )
    ##     # Two arrays (I,J) giving the unique edges, (j,i)'s have been eliminated.
    ##     unique_edges = new_edge_id_mtx.nonzero()
    ##     n_unique_edges = len( unique_edges[0] )
    ##     # Assign indices 1,...,n to the unique edges.
    ##     new_edge_id_mtx.data = np.arange( 1, n_unique_edges+1 )
    ##     # Mapping between new indices and both versions of edges: (i,j),(j,i)
    ##     new_edge_id_mtx = new_edge_id_mtx + new_edge_id_mtx.T

    ##     # In any case, some (j,i)'s don't exist, so create a mask
    ##     # for only the existing edges.
    ##     mask = sparse.csr_matrix( (np.ones(len(I),dtype=int),(I,J)), (n,n) )
    ##     # And extract the ids of the existing edges from new_edge_id_mtx.
    ##     new_edge_ids = ( mask.multiply( new_edge_id_mtx ) ).data - 1

    ##     # The information obtained from sparse matrix operations has
    ##     # a different ordering than the original (I,J); therefore,
    ##     # they need to mapped to the original edge locations/indices.
    ##     edge_indices = sparse.csr_matrix( (np.arange(len(I)),(I,J)), (n,n) ).data

    ##     # Assign the new edge ids to each of (i,j) in (I,J)
    ##     edge_number = np.empty(len(I),dtype=int)
    ##     edge_number[ edge_indices ] = new_edge_ids

    ##     element2edges = np.reshape( edge_number, elem.shape )
    ##     edge2nodes = np.vstack( unique_edges )

    ##     return (element2edges, edge2nodes)


    ## def refine_coarsen(self, markers, data_vectors=[], conforming=False):
    ##     tri = self._triangles
    ##     vtx = self._vertices
    ##     n_elem = tri.shape[1]

    ##     element2edges, edge2nodes = self._compute_geom_info()

    ##     edge2newnode = np.zeros( np.max(element2edges) + 1, dtype=int )
    ##     # Split the marked element into by cutting the ref edge (1st edge).
    ##     edge2newnode[ element2edges[0,markers] ] = 1
    ##     # Alternatively split the marked element into four.
    ##     # edge2newnode[ element2edges[:,markers] ] = 1

    ##     # If conforming, both sides of an edge should be refined, no hanging nodes.
    ##     if conforming: # Iterate until there are no hanging nodes.
    ##         swap = np.array([0.0])
    ##         while len(swap) > 0:
    ##             marked = edge2newnode[ element2edges ]
    ##             swap = np.nonzero( ~marked[0,:] & (marked[1,:] | marked[2,:]) )[0]
    ##             edge2newnode[ element2edges[0,swap] ] = 1
    ##     else:
    ##         marked_elem = markers

    ##     n_old_nodes = vtx.shape[1]
    ##     n_new_nodes = np.sum( edge2newnode )
    ##     # Assign the new node numbers.
    ##     edge2newnode[ edge2newnode != 0 ] = range( n_old_nodes,
    ##                                                 n_old_nodes + n_new_nodes )
    ##     # Calculate the new node coordinates.
    ##     idx = np.nonzero( edge2newnode )[0]
    ##     new_coords = np.empty(( 2, n_old_nodes + n_new_nodes ))
    ##     new_coords[:,0:n_old_nodes] = vtx
    ##     new_coords[:,edge2newnode[idx]] = 0.5 * (vtx[:, edge2nodes[0,idx] ] +
    ##                                              vtx[:, edge2nodes[1,idx] ])

    ##     new_nodes = edge2newnode[ element2edges ]
    ##     marked = (new_nodes != 0) # marked edges are the ones with new nodes

    ##     if conforming:
    ##         none = ~marked[0,:]
    ##         bisec1   = marked[0,:] & ~marked[1,:] & ~marked[2,:]
    ##         bisec12  = marked[0,:] &  marked[1,:] & ~marked[2,:]
    ##         bisec13  = marked[0,:] & ~marked[1,:] &  marked[2,:]
    ##         bisec123 = marked[0,:] &  marked[1,:] &  marked[2,:]
    ##     else: # not conforming
    ##         mask = marked_elem
    ##         none = np.ones( n_elem, dtype=bool )
    ##         none[ mask ] = False
    ##         bisec1   = np.zeros( n_elem, dtype=bool )
    ##         bisec12  = np.zeros( n_elem, dtype=bool )
    ##         bisec13  = np.zeros( n_elem, dtype=bool )
    ##         bisec123 = np.zeros( n_elem, dtype=bool )
    ##         bisec1[mask]   = marked[0,mask] & ~marked[1,mask] & ~marked[2,mask]
    ##         bisec12[mask]  = marked[0,mask] &  marked[1,mask] & ~marked[2,mask]
    ##         bisec13[mask]  = marked[0,mask] & ~marked[1,mask] &  marked[2,mask]
    ##         bisec123[mask] = marked[0,mask] &  marked[1,mask] &  marked[2,mask]

    ##     idx = np.ones( n_elem, dtype=int )
    ##     idx[bisec1]   = 2  # bisec1 creates two new elements
    ##     idx[bisec12]  = 3  # bisec12 creates three new elements
    ##     idx[bisec13]  = 3  # bisec13 creates three new elements
    ##     idx[bisec123] = 4  # bisec123 creates four new elements
    ##     idx = np.hstack(( 0, np.cumsum(idx) ))

    ##     new_elem = np.zeros((3, idx[-1]), dtype=int)

    ##     new_elem[:,idx[none]] = tri[:,none]

    ##     new_elem[:, idx[bisec1] ] = \
    ##                 (tri[ 2, bisec1 ], tri[ 0, bisec1 ], new_nodes[ 0, bisec1 ])
    ##     new_elem[:, 1+idx[bisec1] ] = \
    ##                 (tri[ 1, bisec1 ], tri[ 2, bisec1 ], new_nodes[ 0, bisec1 ])

    ##     new_elem[:, idx[bisec12] ] = \
    ##                 (tri[ 2, bisec12 ],      tri[ 0, bisec12 ], new_nodes[ 0, bisec12 ])
    ##     new_elem[:, 1+idx[bisec12] ] = \
    ##                 (new_nodes[ 0, bisec12 ], tri[ 1, bisec12 ], new_nodes[ 1, bisec12 ])
    ##     new_elem[:, 2+idx[bisec12] ] = \
    ##                 (tri[ 2, bisec12 ], new_nodes[ 0, bisec12 ], new_nodes[ 1, bisec12 ])

    ##     new_elem[:, idx[bisec13] ] = \
    ##                 (new_nodes[ 0, bisec13 ], tri[ 2, bisec13 ], new_nodes[ 2, bisec13 ])
    ##     new_elem[:, 1+idx[bisec13] ] = \
    ##                 (tri[ 0, bisec13 ], new_nodes[ 0, bisec13 ], new_nodes[ 2, bisec13 ])
    ##     new_elem[:, 2+idx[bisec13] ] = \
    ##                 (tri[ 1, bisec13 ],      tri[ 2, bisec13 ], new_nodes[ 0, bisec13 ])

    ##     new_elem[:, idx[bisec123] ] = \
    ##                 (new_nodes[ 0, bisec123 ], tri[ 2, bisec123 ], new_nodes[ 2, bisec123 ])
    ##     new_elem[:, 1+idx[bisec123] ] = \
    ##                 (tri[ 0, bisec123 ], new_nodes[ 0, bisec123 ], new_nodes[ 2, bisec123 ])
    ##     new_elem[:, 2+idx[bisec123] ] = \
    ##                 (new_nodes[ 0, bisec123 ], tri[ 1, bisec123 ], new_nodes[ 1, bisec123 ])
    ##     new_elem[:, 3+idx[bisec123] ] = \
    ##                 (tri[ 2, bisec123 ], new_nodes[ 0, bisec123 ], new_nodes[ 1, bisec123 ])

    ##     self._triangles = new_elem
    ##     self._vertices = new_coords
    ##     self.reset_data()


    ##     # Create new data vectors preserving data on unchanged elements.
    ##     new_vectors = []

    ##     for old_vec in data_vectors:

    ##         new_vec_size = old_vec.shape[0:-1] + (new_elem.shape[1],)
    ##         new_vec = np.empty( new_vec_size )
    ##         new_vec[:] = np.nan

    ##         if new_vec.ndim == 1:
    ##             new_vec[idx[none]] = old_vec[none]
    ##         elif new_vec.ndim == 2:
    ##             new_vec[:,idx[none]] = old_vec[:,none]
    ##         elif new_vec.ndim == 3:
    ##             new_vec[:,:,idx[none]] = old_vec[:,:,none]
    ##         else:
    ##             raise ValueError("Data_vector.ndim > 3 not allowed!")

    ##         new_vectors.append( new_vec )

    ##     return new_vectors


    def _compute_new_nodes(self, marked):
        """Computes new nodes & coordinates for marked edges.

        The marked edges will be split into two edges from midpoint of
        the marked edge. This function only computes the new coordinates
        and the corresponding node ids for the new midpoint nodes.
        The actual splitting and changes to the triangulation data
        structures are done by the caller routine.

        Parameters
        ----------
        marked : tuple of NumPy arrays
            A pair of integer arrays (mark_i, mark_j), which store the
            indices of the marked edges of the elements to be refined.
            mark_i stores the edge number and mark_j stores the element
            index.

        Returns
        -------
        new_nodes : NumPy array
            A Numpy array of integers storing the node id numbers
            for the new nodes of the marked edges.
        new_coords : NumPy array
            A (2,N) dimensional Numpy float array storing the x,y
            coordinates of the new nodes.
        """

        elem = self._triangles
        n_elem = elem.shape[1]
        coords = self._vertices
        n = coords.shape[1]

        mark_i, mark_j = marked
        original_edge_index = mark_i * n_elem + mark_j
        I = elem[ mark_i, mark_j ]
        J = elem[ (mark_i+1) % 3, mark_j ]
        min_IJ = np.minimum(I,J)
        max_IJ = np.maximum(I,J)

        dummy = np.empty( len(I), dtype=int )
        new_node_id_mtx = sparse.csr_matrix( (dummy, (min_IJ,max_IJ)), (n,n) )
        n_new_nodes = new_node_id_mtx.getnnz()
        new_node_id_mtx.data = np.arange( n, n + n_new_nodes )
        I2, J2 = new_node_id_mtx.nonzero()
        new_coords = 0.5 * (coords[:,I2] + coords[:,J2])

        new_node_id_mtx = new_node_id_mtx + new_node_id_mtx.T

        original_edges_mtx = sparse.csr_matrix( (original_edge_index,(I,J)), (n,n) )
        vec_len = len(original_edges_mtx.data)
        mask = sparse.csr_matrix( (np.ones(vec_len, dtype=int),
                                   original_edges_mtx.indices,
                                   original_edges_mtx.indptr ), (n,n) )
        new_node_ids = ( mask.multiply( new_node_id_mtx ) ).data

        edge_indices = original_edges_mtx.data
        index0 = edge_indices / n_elem
        index1 = edge_indices % n_elem
        new_nodes = np.zeros_like( elem )
        new_nodes[ index0, index1 ] = new_node_ids

        return (new_nodes, new_coords)


    def refine_coarsen(self, markers, data_vectors=[], conforming=False):
        if conforming:
            raise ValueError("Does not work for conforming refinement yet!")

        old_elem = self._triangles
        n_old_elem = old_elem.shape[1]

        if markers.dtype == bool:
            marked = markers.nonzero()[0]
        elif markers.dtype == int:
            marked = markers
        else:
            raise ValueError("markers should either be a boolean array or an integer array of element indices!")
        # n_new_elem = len(marked)
        n_new_elem = 3*len(marked)

        # (i,j) indices of primary edges of the marked triangles
        # mark_j = marked
        # mark_i = np.zeros_like(mark_j)
        n_marked = len(marked)
        mark_j = np.hstack(( marked, marked, marked ))
        mark_i = np.empty( 3*n_marked, dtype=int )
        mark_i[0:n_marked] = 0
        mark_i[n_marked:2*n_marked] = 1
        mark_i[2*n_marked:3*n_marked] = 2
        need_to_split = ( mark_i, mark_j ) # edges to be split

        # Compute the new node, their coords and assign to correct elems.
        new_nodes, new_coords = self._compute_new_nodes( need_to_split )
        new_coords = np.hstack( (self._vertices, new_coords) )

        # Create the new element array.
        new_elem = np.empty( (3, n_old_elem + n_new_elem), dtype=int )

        # Copy all the old elements to the new_elem array.
        # The marked locations will be overwritten next.
        new_elem[:,0:n_old_elem] = old_elem

        ## # Overwrite some locations with the firsts of the new element pairs.
        ## new_elem[:,marked] = \
        ##     (old_elem[2,marked], old_elem[0,marked], new_nodes[0,marked])

        ## # The seconds of the new element pairs are added to the end.
        ## new_elem[:,n_old_elem:] = \
        ##     (old_elem[1,marked], old_elem[2,marked], new_nodes[0,marked])

        inc = n_new_elem / 3
        # Overwrite some locations with the first of the four new elements.
        new_elem[:, marked] = \
            (new_nodes[0,marked], old_elem[2,marked], new_nodes[2,marked])
        # The seconds, thirds & fourths are are added to the end.
        new_elem[:, n_old_elem:(n_old_elem + inc)] = \
            (old_elem[0,marked], new_nodes[0,marked], new_nodes[2,marked])
        new_elem[:, (n_old_elem + inc):(n_old_elem + 2*inc)] = \
            (new_nodes[0,marked], old_elem[1,marked], new_nodes[1,marked])
        new_elem[:, (n_old_elem + 2*inc):(n_old_elem + 3*inc)] = \
            (old_elem[2,marked], new_nodes[0,marked], new_nodes[1,marked])

        # Assign the new elements and new coords for the triangulation.
        self._triangles = new_elem
        self._vertices = new_coords
        self.timestamp += 1

        #-----------------------------------------------------------------

        # Indices of the elements added to the triangulation.
        new_indices = np.hstack((marked,
                                 np.arange(n_old_elem, n_old_elem + n_new_elem)))

        # If self._coords and _el_sizes have been precomputed, update them.

        if self._coords is not None:
            missing_coords = self.coords(new_indices)
            coords = []
            for i in range(3):
                coords.append( np.empty((2, new_elem.shape[1])) )
                coords[i][:,0:n_old_elem] = self._coords[i]
                coords[i][:,new_indices] = missing_coords[i]
            self._coords = tuple(coords)

        if self._el_sizes is not None:
            el_sizes = np.empty( new_elem.shape[1] )
            el_sizes[0:n_old_elem] = self._el_sizes
            el_sizes[new_indices] = self.element_sizes(new_indices)
            self._el_sizes = el_sizes

        # !!! In the previous, it is critical that the order of operations !!!
        # !!! is (triangles, vertices), coords, el_sizes.                  !!!

        #--------------------------------------------------------------

        # Data vectors store data associated with elements.
        # Create new data vectors preserving the data on unchanged elements.
        new_vectors = []

        for old_vec in data_vectors:

            new_vec_size = old_vec.shape[0:-1] + (new_elem.shape[1],)
            new_vec = np.empty( new_vec_size )

            if new_vec.ndim == 1:
                new_vec[0:n_old_elem] = old_vec
                new_vec[new_indices] = np.nan

            elif new_vec.ndim == 2:
                new_vec[:,0:n_old_elem] = old_vec
                new_vec[:,new_indices] = np.nan

            elif new_vec.ndim == 3:
                new_vec[:,:,0:n_old_elem] = old_vec
                new_vec[:,:,new_indices] = np.nan
            else:
                raise ValueError("Data_vector.ndim > 3 not allowed!")

            new_vectors.append( new_vec )

        return (new_vectors, new_indices)


    def mark_grid(self, grid, origin=(0.0,0.0), size=(1.0,1.0), value=1):
        x0, x1, x2 = self.coords()

        Domain2d._mark_grid_loop(grid, x0,x1,x2, origin, size, value)


    @jit( nopython = True )
    def _mark_grid_loop(grid, x0,x1,x2, origin, size, value):

        n_tri = x0.shape[1]
        nx, ny = grid.shape
        size_x, size_y = size
        origin_x, origin_y = origin

        inc_x = size_x / (nx - 1.0)
        inc_y = size_y / (ny - 1.0)
        offset_x = origin_x / inc_x
        offset_y = origin_y / inc_y
        domain_min_x = origin_x
        domain_max_x = origin_x + size_x
        domain_min_y = origin_y
        domain_max_y = origin_y + size_y

        for k in range(0,n_tri):
            min_x = min( min(x0[0,k], x1[0,k]), x2[0,k] )
            max_x = max( max(x0[0,k], x1[0,k]), x2[0,k] )
            min_y = min( min(x0[1,k], x1[1,k]), x2[1,k] )
            max_y = max( max(x0[1,k], x1[1,k]), x2[1,k] )

            if max_x < domain_min_x:  continue
            if max_y < domain_min_y:  continue
            if min_x > domain_max_x:  continue
            if min_y > domain_max_y:  continue

            v0_x = x0[0,k];  v0_y = x0[1,k]
            v1_x = x1[0,k] - v0_x;  v1_y = x1[1,k] - v0_y
            v2_x = x2[0,k] - v0_x;  v2_y = x2[1,k] - v0_y

            det_v0_v1 = v0_x*v1_y - v0_y*v1_x
            det_v0_v2 = v0_x*v2_y - v0_y*v2_x
            det_v1_v2 = v1_x*v2_y - v1_y*v2_x

            min_i = max(  0,   np.ceil(  min_x/inc_x - offset_x ) )
            max_i = min( nx-1, np.floor( max_x/inc_x - offset_x ) )
            min_j = max(  0,   np.ceil(  min_y/inc_y - offset_y ) )
            max_j = min( ny-1, np.floor( max_y/inc_y - offset_y ) )

            min_x = domain_min_x + inc_x * min_i
            max_x = domain_min_x + inc_x * max_i
            min_y = domain_min_y + inc_y * min_j
            max_y = domain_min_y + inc_y * max_j

            px = min_x
            for i in range(min_i, max_i):
                py = min_y
                for j in range(min_j, max_j):
                    det_p_v1 = px*v1_y - py*v1_x
                    det_p_v2 = px*v2_y - py*v2_x
                    a =  (det_p_v2 - det_v0_v2) / det_v1_v2
                    b = -(det_p_v1 - det_v0_v1) / det_v1_v2
                    if (a >= 0.0) and (b >= 0.0) and (a+b <= 1.0):
                        grid[i,j] = value
                    py += inc_y
                px += inc_x


    def show(self, tri=None, vtx=None, mask=None, format='k-', factor=1.0):
        import matplotlib.pyplot as plt

        if (tri is None) or (vtx is None):
            tri = self._triangles
            vtx = factor*self._vertices

        if mask is None:
            plt.triplot( vtx[0], vtx[1], format, triangles=tri.T )
        else:
            mask2 = np.ones( tri.shape[1], dtype=boolean ) # Set all to true.
            mask2[mask] = False # Those given by mask will be displayed.
            plt.triplot( vtx[0], vtx[1], format, triangles=tri.T, mask=mask2 )

        plt.axis([ vtx[0].min(), vtx[0].max(), vtx[1].min(), vtx[1].max() ])
        plt.show()
