import numpy as np
import numba
from numba import jit
from numpy import logical_and as np_AND, logical_or as np_OR, logical_not as np_NOT
from numpy import maximum, minimum
from itertools import combinations
from collections import deque
from .curve import Curve, CurveHierarchy


@jit( (numba.float64[:,:], numba.float64[:,:],
       numba.float64[:,:], numba.float64[:,:]),  nopython=True )
def edges_intersect(start0, end0, start1, end1):

    # NOTE: the start nodes are considered to be part of the edges,
    # but the end nodes are excluded. This is taken into consideration
    # in the following checks.

    n_edges1 = start1.shape[1]
    start0_x = start0[0,0]
    start0_y = start0[1,0]
    end0_x = end0[0,0]
    end0_y = end0[1,0]

    check = np.zeros( n_edges1, dtype=np.bool_ )

    # We rotate the coordinate system, so that edge0, (start0, end0),
    # is aligned with the x-axis and the node start0 is translated
    # to the origin. Similarly we translate and rotate edge1.

    edge0_length = np.sqrt( (end0_x - start0_x)**2 + (end0_y - start0_y)**2 )

    cosine = (end0_x - start0_x) / edge0_length
    sine   = (end0_y - start0_y) / edge0_length

    if start0_x < end0_x:
        min_x0 = start0_x;   max_x0 = end0_x
    else:
        max_x0 = start0_x;   min_x0 = end0_x

    if start0_y < end0_y:
        min_y0 = start0_y;   max_y0 = end0_y
    else:
        max_y0 = start0_y;   min_y0 = end0_y

    for k in range(0,n_edges1):

        # We use min/max_x0, min_max_y0, min/max_x1, min_max_y1 to identify
        # the boundary boxes of the two edges. If the two boundary boxes
        # don't intersect, then the edges cannot intersect.

        if start1[0,k] < end1[0,k]:
            min_x1 = start1[0,k];   max_x1 = end1[0,k]
        else:
            max_x1 = start1[0,k];   min_x1 = end1[0,k]

        if start1[1,k] < end1[1,k]:
            min_y1 = start1[1,k];   max_y1 = end1[1,k]
        else:
            max_y1 = start1[1,k];   min_y1 = end1[1,k]

        # After the bounding box check, now the edges might be intersecting.
        # So rotate edge1 to the new coordinate system of edge0.

        if (min_x0 <= max_x1) and (max_x0 >= min_x1) and \
           (min_y0 <= max_y1) and (max_y0 >= min_y1):

            # First rotate the translated y coordinates. If both of them
            # are positive or negative, then edge1 doesn't intersect edge0.
            new_y1 =   -sine  * ( start1[0,k] - start0_x ) \
                     + cosine * ( start1[1,k] - start0_y )
            new_y2 =   -sine  * ( end1[0,k] - start0_x ) \
                     + cosine * ( end1[1,k] - start0_y )

            # Next rotate the translated x coordinates. If both of them are
            # negative or greater than edge0_length, then edge1 doesn't
            # intersect edge0.

            if (new_y1 == 0) and (new_y2 == 0):

                new_x1 = cosine * ( start1[0,k] - start0_x ) \
                         + sine * ( start1[1,k] - start0_y )
                new_x2 = cosine * ( end1[0,k] - start0_x ) \
                         + sine * ( end1[1,k] - start0_y )

                if (0 < new_x2) and (new_x1 <= edge0_length):

                    check[k] = True

            elif ((new_y1 >= 0) or (new_y2 > 0)) and \
                 ((new_y1 <= 0) or (new_y2 < 0)):

                new_x1 = cosine * ( start1[0,k] - start0_x ) \
                         + sine * ( start1[1,k] - start0_y )
                new_x2 = cosine * ( end1[0,k] - start0_x ) \
                         + sine * ( end1[1,k] - start0_y )
                # Compute the intersection point with the new x-axis.
                # If x-coords of the intersection point is between
                # [0, edge0_length), then edges intersect.
                new_x1 -= new_y1 * (new_x2 - new_x1) / (new_y2 - new_y1)

                if (0.0 <= new_x1) and (new_x1 < edge0_length):

                    check[k] = True

    # If we have passed all the checks above, then edge0 and edge1 intersect.
    # So check is true for those edges.

    return check


def compute_edge_intersections(edge0, edge_list):

    start_x, start_y = edge0[0][1]  # coords of first/0/start node
    end_x,   end_y   = edge0[1][1]  # coords of second/1/end node

    edge0_length_sqr = (end_x - start_x)**2 + (end_y - start_y)**2

    cosine = (end_x - start_x) / edge0_length_sqr
    sine   = (end_y - start_y) / edge0_length_sqr

    # Note: edge[0] is start node of edge, edge[0][1] is the coord of
    # the start node, edge[0][1][0] is the x-coord of the start node.
    translated_coords = ( ( edge_id,
                            node1[0] - start_x, node1[1] - start_y,
                            node2[0] - start_x, node2[1] - start_y )
                          for edge_id, node1, node2 in edge_list )

    rotated_coords = ( ( edge_id,
                         cosine * x1 + sine * y1,  -sine * x1 + cosine * y1,
                         cosine * x2 + sine * y2,  -sine * x2 + cosine * y2 )
                       for edge_id, x1,y1,x2,y2 in translated_coords )

    intersection_pts = ( ( edge_id, x1 - y1 * (x2-x1)/(y2-y1) )
                         for edge_id, x1,y1,x2,y2 in rotated_coords )

    # Return a sorted list of pairs (edge object, intersection pt)
    return sorted( intersection_pts, key=lambda p: p[1] )


@jit( nopython = True )
def _edges_in_bounding_box(coords, box):
    # First check all the edges but the last.
    x0, y0 = coords[0,:-1], coords[1,:-1]
    x1, y1 = coords[0,1:],  coords[1,1:]

    edge_indices, = np.nonzero( np_AND( \
                        np_AND( minimum(x0,x1) <= box['max'][0],
                                maximum(x0,x1) >= box['min'][0] ),
                        np_AND( minimum(y0,y1) <= box['max'][1],
                                maximum(y0,y1) >= box['min'][1] ) ))

    # Now check the last edge also. If it is in, add it to node_indices.
    n = coords.shape[1]
    x0, y0 = coords[:,n-1]
    x1, y1 = coords[:,0]
    if min(x0,x1) <= box['max'][0] and max(x0,x1) >= box['min'][0] and \
       min(y0,y1) <= box['max'][1] and max(y0,y1) >= box['min'][1]:
        edge_indices = np.append( edge_indices, n-1 )

    return edge_indices


@jit( nopython = True )
def compute_intersection_of_two_curves(curve1_id, curve1,
                                       curve2_id, curve2,
                                       intersection_list=[],
                                       intersecting_edges={},
                                       edge_info={}):

    bbox1, bbox2 = curve1.bounding_box(), curve2.bounding_box()

    # If the two bounding boxes don't intersect, the curves don't intersect.
    if (bbox1['max'][0] < bbox2['min'][0]) or \
       (bbox1['min'][0] > bbox2['max'][0]) or \
       (bbox1['max'][1] < bbox2['min'][1]) or \
       (bbox1['min'][1] > bbox2['max'][1]):
        return (intersection_list, intersecting_edges, edge_info)

    # Get the coordinate arrays of the two curves.
    coords1, coords2 = curve1.coords(), curve2.coords()
    n1, n2 = coords1.shape[1], coords2.shape[1]

    # Find the indices of the nodes that are in the bounding box of the
    # other curve. Only those edges should be checked for intersection.
    if curve1_id == curve2_id:
        indices1 = np.arange(0,n1-2)
    else: # if curve 1 & curve2 are different curves (not self-intersection)
        indices1 = _edges_in_bounding_box( coords1, bbox2 )
        indices2 = _edges_in_bounding_box( coords2, bbox1 )

    for i1 in indices1:
        i2 = (i1 + 1) % n1

        if curve1_id == curve2_id: # Checking for self-intersections of a curve
            indices2 = np.arange(i1+2,n2)

        # nodes1 = ( coords1[:,i1].reshape(2,1), coords1[:,i2].reshape(2,1) )
        edge1_start = coords1[:,i1] #.reshape(2,1)
        edge1_end   = coords1[:,i2] #.reshape(2,1)
        edge1_id = (curve1_id, i1)
        edge1 = ( ( (curve1_id, i1), edge1_start.reshape(2,1) ),
                  ( (curve1_id, i2), edge1_end.reshape(2,1)   ) )
        # edge1 = ( ( (curve1_id, i1), nodes1[0] ),
        #           ( (curve1_id, i2), nodes1[1] ) )
        edge_info[ edge1_id ] = edge1

        # nodes2 = ( coords2[:,indices2], coords2[:,(indices2+1)%n2] )
        edges2_start = coords2[:, indices2 ]
        edges2_end   = coords2[:,(indices2+1)%n2 ]
        check = edges_intersect( edge1_start, edge1_end, edges2_start, edges2_end )
        # check = edges_intersect( nodes1, nodes2 )

        # Iterate over the indices j of the edges of the 2nd curve
        # that are found to intersect with edge1 (of curve1).
        for j1 in indices2[check]:
            j2 = (j1 + 1) % n2
            edge2_id = (curve2_id, j1)
            edge2 = ( ( (curve2_id, j1), coords2[:,j1].reshape(2,1) ) ,
                      ( (curve2_id, j2), coords2[:,j2].reshape(2,1) ) )
            # Append the intersection to the intersection list.
            intersection_list.append( (edge1, edge2) )
            # Add edge2 to intersecting elements list of edge1 (curve1_id,i1).
            edge_set = intersecting_edges.setdefault( edge1_id, set() )
            edge_set.add( edge2_id )
            # Add edge1 to intersecting elements list of edge2 (curve2_id,j1).
            edge_set = intersecting_edges.setdefault( edge2_id, set() )
            edge_set.add( edge1_id )
            # Also store the info for edge2: start, end nodes, coords.
            edge_info[ edge2_id ] = edge2

    return (intersection_list, intersecting_edges, edge_info)


## def compute_intersection_of_two_curves(curve1_id, curve1,
##                                        curve2_id, curve2,
##                                        intersection_list=[],
##                                        intersecting_edges={},
##                                        edge_info={}):

##     bbox1, bbox2 = curve1.bounding_box(), curve2.bounding_box()
##     b1_min_x, b1_min_y = bbox1['min']
##     b1_max_x, b1_max_y = bbox1['max']
##     b2_min_x, b2_min_y = bbox2['min']
##     b2_max_x, b2_max_y = bbox2['max']

##     # If the two boundary boxes don't intersect, the curves don't intersect.
##     if (b1_max_x < b2_min_x) or (b1_min_x > b2_max_x) or \
##        (b1_max_y < b2_min_y) or (b1_min_y > b2_max_y):
##         return (intersection_list, intersecting_edges, edge_info)

##     # Get the coordinate arrays of the two curves.
##     coords1, coords2 = curve1.coords(), curve2.coords()
##     n1, n2 = coords1.shape[1], coords2.shape[1]

##     # Fast C code the compute the intersections.
##     function_code = \
##     """
##     PyObject *construct_edge(PyObject *edge_info, int curve_id, int i1, int i2,
##                              double x1, double y1, double x2, double y2)
##     {
##         long int pair_dim[] = {2,1};
##         PyObject *coord_pair, *node1_id, *node2_id, *node1, *node2, *edge;

##         node1_id = PyTuple_Pack( 2, PyInt_FromLong(curve_id), PyInt_FromLong(i1));

##         coord_pair = PyArray_SimpleNew( 2, pair_dim, PyArray_DOUBLE );
##         *((double *) PyArray_GETPTR2( coord_pair, 0, 0 )) = x1;
##         *((double *) PyArray_GETPTR2( coord_pair, 0, 1 )) = y1;

##         node1 = PyTuple_Pack( 2, node1_id, coord_pair );

##         node2_id = PyTuple_Pack( 2, PyInt_FromLong(curve_id), PyInt_FromLong(i2));

##         coord_pair = PyArray_SimpleNew( 2, pair_dim, PyArray_DOUBLE );
##         *((double *) PyArray_GETPTR2( coord_pair, 0, 0 )) = x2;
##         *((double *) PyArray_GETPTR2( coord_pair, 0, 1 )) = y2;

##         node2 = PyTuple_Pack( 2, node2_id, coord_pair );

##         edge = PyTuple_Pack( 2, node1, node2 );

##         PyDict_SetItem( edge_info, node1_id, edge );

##         return edge;
##     }

##     int edge_in_bounding_box(double x1, double y1, double x2, double y2,
##                           double min_x, double max_x, double min_y, double max_y)
##     {
##         if ((x1 < min_x) && (x2 < min_x))  return 0;
##         if ((x1 > max_x) && (x2 > max_x))  return 0;
##         if ((y1 < min_y) && (y2 < min_y))  return 0;
##         if ((y1 > max_y) && (y2 > max_y))  return 0;
##         return 1;
##     }

##     int edges_intersect(double x3, double y3, double x4, double y4,
##                         double x1, double y1,
##                         double min_x1, double max_x1, double min_y1, double max_y1,
##                         double cosine, double sine, double edge1_length)
##     {
##         double min_x2, max_x2, min_y2, max_y2, new_x1, new_x2, new_y1, new_y2;

##         if (x3 < x4) { min_x2 = x3;  max_x2 = x4; }
##             else     { min_x2 = x4;  max_x2 = x3; }
##         if (y3 < y4) { min_y2 = y3;  max_y2 = y4; }
##             else     { min_y2 = y4;  max_y2 = y3; }

##         if ((min_x1 <= max_x2) && (max_x1 >= min_x2) &&
##             (min_y1 <= max_y2) && (max_y1 >= min_y2))
##         {
##             new_y1 = -sine * (x3 - x1) + cosine * (y3 - y1);
##             new_y2 = -sine * (x4 - x1) + cosine * (y4 - y1);

##             if (((new_y1 >= 0) || (new_y2 > 0)) && ((new_y1 <= 0) || (new_y2 < 0)))
##             {
##                 new_x1 = cosine * (x3 - x1) + sine * (y3 - y1);
##                 new_x2 = cosine * (x4 - x1) + sine * (y4 - y1);

##                 new_x1 -= new_y1 * (new_x2 - new_x1) / (new_y2 - new_y1);

##                 if ((0.0 <= new_x1) && (new_x1 < edge1_length))  return 1;
##             }
##         }
##         return 0;
##     }
##     """

##     code = \
##     """
##     PyObject *edge1, *edge1_id, *edge2, *edge2_id, *intersection, *edge_set;
##     int edges_can_intersect, found_intersection;
##     double x1, y1, x2, y2, x3, y3, x4, y4, min_x1, max_x1, min_y1, max_y1;
##     double cosine, sine, edge1_length;
##     int i, i2, j, j2, j_start, j_end;
##     double box1_min_x = b1_min_x, box1_max_x = b1_max_x;
##     double box1_min_y = b1_min_y, box1_max_y = b1_max_y;
##     double box2_min_x = b2_min_x, box2_max_x = b2_max_x;
##     double box2_min_y = b2_min_y, box2_max_y = b2_max_y;
##     int curve1_id = c1_id, curve2_id = c2_id;


##     // First find the indices of the edges of the second curve
##     // that are in the bounding box of the first curve.

##     std::list<int> j_list;
##     int n_j_indices, *j_indices = NULL;

##     if (curve1_id == curve2_id)
##     {
##         n_j_indices = n2;
##         j_indices = new int[n2];
##         for (int j = 0; j < n2; j++)  j_indices[j] = j;
##     }
##     else // curve1_id != curve2_id, so we need to check
##     {
##         for (j = 0; j < n2; j++) {
##             j2 = (j + 1) % n2;
##             x3 = coords2(0,j);    y3 = coords2(1,j);
##             x4 = coords2(0,j2);   y4 = coords2(1,j2);
##             if (edge_in_bounding_box( x3, y3, x4, y4,
##                            box1_min_x, box1_max_x, box1_min_y, box1_max_y ))
##                 j_list.push_back( j );
##         }
##         n_j_indices = j_list.size();
##         j_indices = new int[n_j_indices];
##         std::list<int>::iterator iter;
##         for (j=0, iter=j_list.begin(); iter != j_list.end(); iter++, j++)
##              j_indices[j] = *iter;
##     }

##     // Now check for intersections of all edge pairs.

##     for (i = 0; i < n1; i++)
##     {
##         // Get the coordinates of the current edge of curve1.
##         i2 = (i + 1) % n1;
##         x1 = coords1(0,i);    y1 = coords1(1,i);
##         x2 = coords1(0,i2);   y2 = coords1(1,i2);

##         // Check if the edge of curve1 is in the bounding box of curve2.
##         if (curve1_id == curve2_id)
##             edges_can_intersect = 1;
##         else
##             edges_can_intersect = edge_in_bounding_box( x1, y1, x2, y2,
##                                   box2_min_x, box2_max_x, box2_min_y, box2_max_y );

##         if (edges_can_intersect)
##         {
##             if (x1 < x2) { min_x1 = x1;  max_x1 = x2; }
##                 else     { min_x1 = x2;  max_x1 = x1; }
##             if (y1 < y2) { min_y1 = y1;  max_y1 = y2; }
##                 else     { min_y1 = y2;  max_y1 = y1; }
##             edge1_length = sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
##             cosine = (x2 - x1) / edge1_length;
##             sine   = (y2 - y1) / edge1_length;

##             j_start = (curve1_id == curve2_id) ? (i+2) : 0;
##             j_end = (curve1_id == curve2_id) ? (n2-2) : (n_j_indices-1);
##             for (int j0 = j_start; j0 <= j_end; j0++)
##             {
##                 j = j_indices[j0];

##                 // Get the coordinates of the current edge of curve2.
##                 j2 = (j + 1) % n2;
##                 x3 = coords2(0,j);    y3 = coords2(1,j);
##                 x4 = coords2(0,j2);   y4 = coords2(1,j2);

##                 // Check for intersection of the edges from curve 1 and curve 2.
##                 found_intersection = edges_intersect( x3, y3, x4, y4, x1, y1,
##                                                  min_x1, max_x1, min_y1, max_y1,
##                                                  cosine, sine, edge1_length );

##                 if (found_intersection)
##                 {
##                     // Construct edge1 and edge2 tuples, define intersection tuple.
##                     if (edge1 == NULL){
##                         edge1 = construct_edge( py_edge_info, curve1_id,
##                                                 i,i2,x1,y1,x2,y2);
##                         edge1_id = PyTuple_GetItem( PyTuple_GetItem( edge1, 0 ), 0 );
##                     }
##                     edge2 = construct_edge(py_edge_info,curve2_id,j,j2,x3,y3,x4,y4);
##                     edge2_id = PyTuple_GetItem( PyTuple_GetItem( edge2, 0 ), 0 );

##                     // Set intersection = (edge1,edge2), add to intersection_list.
##                     intersection = PyTuple_Pack( 2, edge1, edge2 );
##                     PyList_Append( py_intersection_list, intersection );

##                     // Add edge2 to intersecting_edges set of edge1.
##                     if (!PyDict_Contains( py_intersecting_edges, edge1_id ))
##                         PyDict_SetItem( py_intersecting_edges,
##                                         edge1_id, PySet_New(NULL) );

##                     edge_set = PyDict_GetItem( py_intersecting_edges, edge1_id );
##                     PySet_Add( edge_set, edge2_id );

##                     // Add edge1 to intersecting_edges set of edge2.
##                     if (!PyDict_Contains( py_intersecting_edges, edge2_id ))
##                         PyDict_SetItem( py_intersecting_edges,
##                                         edge2_id, PySet_New(NULL) );

##                     edge_set = PyDict_GetItem( py_intersecting_edges, edge2_id );
##                     PySet_Add( edge_set, edge1_id );
##                 }
##             }
##         }
##         edge1 = NULL;
##     }
##     if (j_indices != NULL)  delete[] j_indices;
##     """

##     c1_id, c2_id = curve1_id, curve2_id

##     weave.inline( code, ['c1_id','c2_id','coords1','coords2','n1','n2',
##                          'b1_min_x', 'b1_max_x', 'b1_min_y', 'b1_max_y',
##                          'b2_min_x', 'b2_max_x', 'b2_min_y', 'b2_max_y',
##                          'intersection_list','intersecting_edges','edge_info'],
##                   support_code=function_code, headers=['<list>'],
##                   type_converters=converters.blitz, compiler='gcc' )

##     return (intersection_list, intersecting_edges, edge_info)


def compute_intersections(curves):

    curve_list = curves.curve_list()
    topmost_curves, parent, children = curves.curve_hierarchy()

    intersection_list = []
    intersecting_edges = {}
    edge_info = {}

    # Check for self-intersections of each curve (by itself).
    for curve_id, curve in enumerate( curve_list ):
        compute_intersection_of_two_curves( curve_id, curve, curve_id, curve,
                                            intersection_list, intersecting_edges,
                                            edge_info )

    # Check for intersections of all topmost curves.
    for curve1_id, curve2_id in combinations( topmost_curves, 2 ):
        compute_intersection_of_two_curves( curve1_id, curve_list[ curve1_id ],
                                            curve2_id, curve_list[ curve2_id ],
                                            intersection_list, intersecting_edges,
                                            edge_info )

    # Check for the intersections between all parents and their children,
    # also for intersections among its children (but not grandchildren).
    parent_list = deque( topmost_curves )
    while len( parent_list ) > 0:
        curve_id = parent_list.popleft()
        curve_child_set = children[ curve_id ]
        parent_list.extend( curve_child_set )

        for child in curve_child_set:
            compute_intersection_of_two_curves( curve_id, curve_list[curve_id],
                                                child, curve_list[child],
                                                intersection_list,
                                                intersecting_edges,
                                                edge_info )

        for child1, child2 in combinations( curve_child_set, 2 ):
            compute_intersection_of_two_curves( child1, curve_list[child1],
                                                child2, curve_list[child2],
                                                intersection_list,
                                                intersecting_edges,
                                                edge_info )

    return (intersection_list, intersecting_edges, edge_info)


def _curve_inside_boundary(curve, boundary):

    bounding_box = boundary.bounding_box()
    min_x, min_y = bounding_box['min']
    max_x, max_y = bounding_box['max']

    bounding_box = curve.bounding_box()
    curve_min_x, curve_min_y = bounding_box['min']
    curve_max_x, curve_max_y = bounding_box['max']

    if (min_x <= curve_min_x) and (curve_max_x <= max_x) and \
       (min_y <= curve_min_y) and (curve_max_y <= max_y):
        return True
    else:
        return False


def _x_boundary_intersections(curve_id, curve, bdary_edge, new_node_id):
    """Computes the intersections of a curve with a given x-aligned edge.

    Parameters
    ----------
    curve_id
    curve
    bdary_edge
    new_node_id

    Returns
    -------
    intersection_info

    """

    coords = curve.coords()
    x,y = coords
    n = curve.size()
    x0,y0 = bdary_edge[0][1].flatten() # coord of the first node
    x1,y1 = bdary_edge[1][1].flatten() # coord of the second node
    bdary_curve_id = bdary_edge[0][0][0] # 1st part of (1st) id part of 1st node


    # First check for outgoing edges of the curve, possibly crossing
    # the bottom boundary, with start node below, end node above.

    i1 = np.nonzero( y <= y0 )[0] # indices with y-coord of start node below y0
    i2 = (i1 + 1) % n   # indices of the end nodes of the candidate edges
    check = y0 < y[i2]  # check if end nodes are above the bottom boundary
    i1, i2 = i1[check], i2[check]  # proceed with the indices that check

    # Compute the x-coord of the intersection point of the crossing edges.
    intersection_x = x[i1] + (x[i2] - x[i1]) * (y0 - y[i1]) / (y[i2] - y[i1])
    # Check if the intersection x is btw the start & end of bdary edge.
    if x0 <= x1:
        check = np_AND( x0 <= intersection_x, intersection_x <  x1 )
    else: # x1 < x0
        check = np_AND( x1 <  intersection_x, intersection_x <= x0 )
    # Get the indices and the intersection points that pass the check.
    indices = list( i1[check] )
    intersection_pts = list( intersection_x[check] )


    # Now check for incoming edges of the curve, possibly crossing
    # the bottom boundary, with start node above, end node below.

    i1 = np.nonzero( y0 <= y )[0] # indices with y-coord of start node above y0
    i2 = (i1 + 1) % n   # indices of the end nodes of the candidate edges
    check = y[i2] < y0  # check if end nodes are below the bottom boundary
    i1, i2 = i1[check], i2[check]  # proceed with the indices that check

    # Compute the x-coord of the intersection point of the crossing edges.
    intersection_x = x[i1] + (x[i2] - x[i1]) * (y0 - y[i1]) / (y[i2] - y[i1])
    # Check if the intersection x is btw the start & end of bdary edge.
    if x0 <= x1:
        check = np_AND( x0 <= intersection_x, intersection_x <  x1 )
    else: # x1 < x0
        check = np_AND( x1 <  intersection_x, intersection_x <= x0 )
    # Get the indices and the intersection points that pass the check.
    indices.extend( i1[check] )
    intersection_pts.extend( intersection_x[check] )


    # Use the sorted list of intersection points and intersecting edge
    # indices to create lists of new nodes on the boundary edge and
    # the corresponding intersecting curve edges.
    intersections = []
    for i1, x in zip( indices, intersection_pts ):
        # Create new node = ((curve_id,node_id), node_coord) for bdary edge.
        bdary_node = ( (bdary_curve_id, new_node_id), np.array([[x],[y0]]) )
        # Create a representation = (node0, node1) for the intersecting
        # edge of the curve.
        i2 = (i1 + 1) % n
        curve_edge = ( ((curve_id, i1), coords[:,i1].reshape((2,1))),
                       ((curve_id, i2), coords[:,i2].reshape((2,1))) )
        intersections.append( (x, bdary_node, curve_edge) )
        new_node_id = new_node_id + 1

    return (intersections, new_node_id)


def _y_boundary_intersections(curve_id, curve, bdary_edge, new_node_id):
    """Computes the intersections of a curve with a given y-aligned edge.

    Parameters
    ----------
    curve_id
    curve
    bdary_edge
    new_node_id

    Returns
    -------
    intersection_info

    """

    coords = curve.coords()
    x,y = coords
    n = curve.size()
    x0,y0 = bdary_edge[0][1].flatten() # coord of the first node
    x1,y1 = bdary_edge[1][1].flatten() # coord of the second node
    bdary_curve_id = bdary_edge[0][0][0] # 1st part of (1st) id part of 1st node


    # First check for outgoing edges of the curve, possibly crossing
    # the bottom boundary, with start node below, end node above.

    i1 = np.nonzero( x <= x0 )[0] # indices with x-coord of start node below x0
    i2 = (i1 + 1) % n   # indices of the end nodes of the candidate edges
    check = x0 < x[i2]  # check if end nodes are above the bottom boundary
    i1, i2 = i1[check], i2[check]  # proceed with the indices that check

    # Compute the y-coord of the intersection point of the crossing edges.
    intersection_y = y[i1] + (y[i2] - y[i1]) * (x0 - x[i1]) / (x[i2] - x[i1])
    # Check if the intersection x is btw the start & end of bdary edge.
    if y0 <= y1:
        check = np_AND( y0 <= intersection_y, intersection_y <  y1 )
    else: # y1 < y0
        check = np_AND( y1 <  intersection_y, intersection_y <= y0 )
    # Get the indices and the intersection points that pass the check.
    indices = list( i1[check] )
    intersection_pts = list( intersection_y[check] )


    # Now check for incoming edges of the curve, possibly crossing
    # the bottom boundary, with start node above, end node below.

    i1 = np.nonzero( x0 <= x )[0] # indices with x-coord of start node above x0
    i2 = (i1 + 1) % n   # indices of the end nodes of the candidate edges
    check = x[i2] < x0  # check if end nodes are below the bottom boundary
    i1, i2 = i1[check], i2[check]  # proceed with the indices that check

    # Compute the y-coord of the intersection point of the crossing edges.
    intersection_y = y[i1] + (y[i2] - y[i1]) * (x0 - x[i1]) / (x[i2] - x[i1])
    # Check if the intersection y is btw the start & end of bdary edge.
    if y0 <= y1:
        check = np_AND( y0 <= intersection_y, intersection_y <  y1 )
    else: # y1 < y0
        check = np_AND( y1 <  intersection_y, intersection_y <= y0 )
    # Get the indices and the intersection points that pass the check.
    indices.extend( i1[check] )
    intersection_pts.extend( intersection_y[check] )


    # Use the sorted list of intersection points and intersecting edge
    # indices to create lists of new nodes on the boundary edge and
    # the corresponding intersecting curve edges.
    intersections = []
    for i1, y in zip( indices, intersection_pts ):
        # Create new node = ((curve_id,node_id), node_coord) for bdary edge.
        bdary_node = ( (bdary_curve_id, new_node_id), np.array([[x0],[y]]) )
        # Create a representation = (node0, node1) for the intersecting
        # edge of the curve.
        i2 = (i1 + 1) % n
        curve_edge = ( ((curve_id, i1), coords[:,i1].reshape((2,1))),
                       ((curve_id, i2), coords[:,i2].reshape((2,1))) )
        intersections.append( (y, bdary_node, curve_edge) )
        new_node_id = new_node_id + 1

    return (intersections, new_node_id)


def compute_boundary_intersections(curve_list, boundary):
    """Computes the intersections of given curve list and boundary curve.

    Given a list of Curve objects and a boundary Curve object, this
    function computes and returns the intersections of the curves with
    the boundary. The boundary is rectangular; therefore, the intersections
    are returned for each of bottom, top, left and right edges of the
    boundary. Namely the intersections list can be of size zero to four.

    Parameters
    ----------
    curve_list : list of Curve objects,
        Intersections of these Curve objects with the boundary will be
        computed.
    boundary : Curve object
        A Curve object defining the boundary. The boundary should be
        rectangular; therefore, the curve should consist of four nodes
        and four axis-aligned edges.

    Returns
    -------
    intersections : list of tuples
        A list intersection triples, each storing the intersections
        with one edge of the curve. For example, the intersection triple
        for the bottom edge of the boundary would be in the form:
            ( bottom_edge, new_nodes, intersecting_edges )
        where intersecting_edges is a list of curve edges that intersect
        the boundary bottom edge at the new node locations.
        New boundary nodes are created for each intersecting edge
        and are stored as list in new_nodes.
        Each edge is a pair of nodes, each node is a pair of node_id
        and a 2x1 NumPy array of node coordinates, each node_id is
        a pair of integers, the curve_id and the curve_node_index.
            bottom_edge = ( bottom_node0, bottom_node1 )
            bottom_node0 = ( (boundary_curve_id, node_id), node)

    """

    # Identify the corner coordinates of the boundary curve.
    bdary_curve_id = len(curve_list) # 0,...,n-1 in the list, n is extra.
    boundary_box = boundary.bounding_box()
    min_x, min_y = boundary_box['min']
    max_x, max_y = boundary_box['max']

    # Create the representations for the bottom, top, left, right edges of
    # the boundary. These representations will be used with the intersections.
    node0, node1, node2, node3 = (bdary_curve_id, 0), (bdary_curve_id, 1), \
                                 (bdary_curve_id, 2), (bdary_curve_id, 3)
    if boundary.orientation() > 0: # counter-clockwise
        coords = np.array( [[min_x, max_x, max_x, min_x],
                            [min_y, min_y, max_y, max_y]] )
        right_edge = ( (node1, coords[:,1].reshape((2,1))),
                       (node2, coords[:,2].reshape((2,1))) )
        left_edge  = ( (node3, coords[:,3].reshape((2,1))),
                       (node0, coords[:,0].reshape((2,1))) )
    else: # clockwise ordering of nodes
        coords = np.array( [[max_x, min_x, min_x, max_x],
                            [min_y, min_y, max_y, max_y]] )
        left_edge  = ( (node1, coords[:,1].reshape((2,1))),
                       (node2, coords[:,2].reshape((2,1))) )
        right_edge = ( (node3, coords[:,3].reshape((2,1))),
                       (node0, coords[:,0].reshape((2,1))) )
    bottom_edge = ( (node0, coords[:,0].reshape((2,1))),
                    (node1, coords[:,1].reshape((2,1))) )
    top_edge    = ( (node2, coords[:,2].reshape((2,1))),
                    (node3, coords[:,3].reshape((2,1))) )
    boundary.set_geometry( coords ) # to make sure nodes are in the right order


    # List of the boundary edges, corresponding intersection detection funcs
    # and boundary intersection lists to simplify some of the following code.
    intersection_info = [ (bottom_edge, _x_boundary_intersections, []),
                          (top_edge,    _x_boundary_intersections, []),
                          (left_edge,   _y_boundary_intersections, []),
                          (right_edge,  _y_boundary_intersections, []) ]


    # The actual intersection detection code:
    # Each curve is checked to see if its bounding box intersects the
    # boundary curve. If the bounding box intersects, then the intersection
    # detection code of the boundary edge is executed.
    new_node_id = 4  # Boundary has nodes 0,1,2,3; next new node is 4.
    outside_curve_set = set()
    for curve_id, curve in enumerate(curve_list):

        if not _curve_inside_boundary( curve, boundary ):

            found_intersections = False
            for bdary_edge, intersection_func, found_list in intersection_info:
                intersections, new_node_id = intersection_func( curve_id, curve,
                                                         bdary_edge, new_node_id )
                if len(intersections) > 0:
                    found_intersections = True
                    found_list.extend( intersections )

            if not found_intersections:
                outside_curve_set.add( curve_id )


    # The intersections for each boundary edge have been computed.
    # Sort those of each bdary edge with respect location on edge
    # from start node to node. Then add them to final list.
    key_func = lambda triple: triple[0] # Key function for sorting
    all_intersections = []
    for bdary_edge, func, intersections in intersection_info:
        # If the boundary edge has intersections, sort and add the final list.
        if len(intersections) > 0:
            # If start node has x (or y) higher than end node, reverse sorting.
            reverse_flag = np.sum( bdary_edge[1][1] - bdary_edge[0][1] ) < 0
            intersections.sort( key=key_func, reverse=reverse_flag )
            # Extract two distinct lists of new boundary nodes and intersecting
            # curve edges from intersection triplets (pt,bdary_node,curve_edge)
            bdary_nodes = [ i[1] for i in intersections ]
            curve_edges = [ i[2] for i in intersections ]
            all_intersections.append( (bdary_edge, bdary_nodes, curve_edges) )

    return (all_intersections, outside_curve_set)


def refine_intersecting_edges(curve_list, intersection_list,
                              intersecting_edge_sets, edge_info):

    # From the intersection list, leave out the intersections whose edges
    # have multiple intersections and store the single intersection edges
    # in new_intersection_list. The one left out will be replaced with
    # the new intersections that will be computed by the code below.
    new_intersection_list = [ (e0,e1) for e0,e1 in intersection_list
                              if  (len( intersecting_edge_sets[ e0[0][0] ] ) == 1)
                              and (len( intersecting_edge_sets[ e1[0][0] ] ) == 1) ]
    # refine_edges[edge_id] gives the list of edges that intersect 'edge'
    # Note that edge_pair = (edge0,edge1), edge0 = (node0,node1)
    # and node0 = ((curve_id,node_id), coord).
    # Therefore, edge0_id = (curve_id, node0_id) = edge_pair[0][0][0]

    # If no intersection has been removed/filtered out by the previous
    # line, then return the intersection list as it is.
    if len( new_intersection_list ) == len( intersection_list ):
        return new_intersection_list

    # Otherwise continue to process the edges with multiple intersections.

    # The dictionary intersecting_edge_sets keeps the edges and the
    # corresponding sets of edges that intersect them. From these
    # entries, we keep the ones that have intersecting edge sets
    # longer than one (i.e. they are intersected a multiple number
    # of times). These edges need to be refined, so that each edge
    # has only one intersection.
    refine_edges = [ (edge0, intersecting_edges)
                     for edge0, intersecting_edges
                     in intersecting_edge_sets.iteritems()
                     if len( intersecting_edges ) > 1 ]

    # Now iterate through all the edges in refine_edges and
    # split them into pieces of single intersections.

    curve_new_node = {} # To store the next new node id for each curve
    for edge0_id, intersecting_edges in refine_edges:

        # Get the data for edge0
        curve0_id = edge0_id[0] # b/c edge_id = (curve_id, edge_start_node)
        curve0 = curve_list[ curve0_id ]
        edge0  = edge_info[ edge0_id ]
        start_node,  end_node  = edge0
        start_coord, end_coord = start_node[1], end_node[1]
        tangent_vector = end_coord - start_coord

        # Obtain the intersection points of the intersecting edges
        # assuming a local coordinate system [0,1) between the start
        # and end of the edge.
        # For this, create triplets of (edge_id, start_coord, end_coord)
        # and input to compute_edge_intersections.
        intersecting_edge_info = ( ( edge,  # edge_id = (curve_id,start_node)
                                     edge_info[edge][0][1],  # = start coord
                                     edge_info[edge][1][1] ) # = end coord
                                   for edge in intersecting_edges )

        intersection_pts = compute_edge_intersections( edge0,
                                                       intersecting_edge_info )

        for k, (edge1_id, pt) in enumerate( intersection_pts[:-1] ):

            # Break edge0 into two edges between pt & pt2.
            pt2 = intersection_pts[k+1][1]
            new_node_coord = start_coord + 0.5*(pt+pt2) * tangent_vector
            new_node_id = curve_new_node.setdefault( curve0_id, curve0.size() )
            new_node = ( (curve0_id, new_node_id), new_node_coord )
            # This creates a new_edge intersecting edge1.
            new_edge_id = start_node[0]
            new_edge = ( start_node, new_node )

            # If edge1 is intersected by edge0, add the intersection
            # (new_edge,edge1) to the intersection_list.
            # If edge1 is intersected by multiple edges (similar to edge0),
            # then we need to split edge1 into pieces too, so we update
            # the intersecting_edges set of edge1, remove edge0 and add
            # new_edge.
            # The intersection of new_edge and the right piece of edge1
            # will be added to the new_intersection_list when we are
            # processing edge1 in one of the next iterations.

            if len( intersecting_edge_sets[ edge1_id ] ) == 1:
                edge1 = edge_info[ edge1_id ]
                new_intersection_list.append( (new_edge, edge1) )

            else: # edge1 also has multiple intersecting edges
                intersecting_edge_sets[ edge1_id ].remove( edge0_id )
                intersecting_edge_sets[ edge1_id ].add( new_edge_id )
                intersecting_edge_sets[ new_edge_id ] = set([ edge1_id ])
                edge_info[ new_edge_id ] = new_edge

            start_node = new_node
            curve_new_node[ curve0_id ] += 1

        # Process the last intersection point, which is a special case
        # because the new edge starts at new node and ends at original
        # end node.
        edge1_id = intersection_pts[-1][0] # = (curve1_id, start_node_id)
        new_edge = ( new_node, end_node )
        new_edge_id = new_node[0]

        if len( intersecting_edge_sets[ edge1_id ] ) == 1:
            edge1 = edge_info[ edge1_id ]
            new_intersection_list.append( (new_edge, edge1) )

        else: # edge1 also has multiple intersecting edges
            intersecting_edge_sets[ edge1_id ].remove( edge0_id )
            intersecting_edge_sets[ edge1_id ].add( new_edge_id )
            intersecting_edge_sets[ new_edge_id ] = set([ edge1_id ])
            edge_info[ new_edge_id ] = new_edge

    return new_intersection_list


def _update_multiple_boundary_intersections(intersection1, intersection_list,
                                    intersection0_index, curve_new_node_id):

    (bdary_edge1, curve_edge0), (curve_edge1, bdary_edge0) = intersection1
    start_node  = curve_edge0[0]
    end_node    = curve_edge1[1]
    start_coord = start_node[1]
    end_coord   = end_node[1]
    edge_length = np.sum( (end_coord - start_coord)**2 )**0.5
    intersection_node1 = curve_edge0[1] # = curve_edge1[0]
    s1 = np.sum( (intersection_node1[1] - start_coord)**2 )**0.5 / edge_length

    intersection0 = intersection_list[ intersection0_index:intersection0_index+2 ]
    (bdary_edge1, curve_edge0), (curve_edge1, bdary_edge0) = intersection0
    intersection_node0 = curve_edge0[1] # = curve_edge1[0]
    s0 = np.sum( (intersection_node0[1] - start_coord)**2 )**0.5 / edge_length

    mid_node_coord = 0.5 * (intersection_node0[1] + intersection_node1[1])
    curve_id = curve_edge0[0][0][0]
    mid_node_id = curve_new_node_id[ curve_id ]
    curve_new_node_id[ curve_id ] = mid_node_id + 1
    mid_node = ( (curve_id, mid_node_id), mid_node_coord )

    if s1 < s0:
        intersection0, intersection1 = intersection1, intersection0
        intersection_node0, intersection_node1 = \
                            intersection_node1, intersection_node0

    edge0 = (start_node, intersection_node0)
    edge1 = (intersection_node0, mid_node)
    edge2 = (mid_node, intersection_node1)
    edge3 = (intersection_node1, end_node)

    (bdary_edge1, curve_edge0), (curve_edge1, bdary_edge0) = intersection0
    intersection0 = (bdary_edge1, edge0), (edge1, bdary_edge0)

    (bdary_edge1, curve_edge0), (curve_edge1, bdary_edge0) = intersection1
    intersection1 = (bdary_edge1, edge2), (edge3, bdary_edge0)

    intersection_list[ intersection0_index ] = intersection0[0]
    intersection_list[ intersection0_index+1 ] = intersection0[1]
    intersection_list.append( intersection1[0] )
    intersection_list.append( intersection1[1] )


def _nodes_coincide(node0,node1):
    # Coincide if |x of node0 - x of node1| + |y of node0 - y of node1| == 0
    coord_difference = abs(node0[1][0] - node1[1][0]) + \
                       abs(node0[1][1] - node1[1][1])
    return (coord_difference == 0.0)


def _edge_pair_for_special_bdary_intersection(curve_edge, curve_list):
    curve_id, start_node_id = curve_edge[0][0]
    n = curve_list[curve_id].size()
    prev_node_id = (start_node_id-1) % n
    coords = curve_list[curve_id].coords()
    prev_node_coord = coords[:,prev_node_id].reshape((2,1))
    prev_node = ((curve_id, prev_node_id), prev_node_coord)
    prev_curve_edge = (prev_node, curve_edge[0])
    return (prev_curve_edge, curve_edge)


def refine_boundary_intersections(curve_list, intersections):

    curve_new_node_id = [ curve.size() for curve in curve_list ]

    new_intersections = []
    intersection_list_index = {} # Stores the index of a previous intersection
                                 # of a curve edge defined by start node id.
    list_index = 0
    for intersection in intersections:
        bdary_edge, new_bdary_nodes, curve_edges = intersection
        bdary_start_node, bdary_final_node = bdary_edge
        new_bdary_nodes.append( bdary_final_node )

        bdary_edge0 = (bdary_start_node, new_bdary_nodes[0])
        for k, (curve_edge, bdary_mid_node) in enumerate(zip( curve_edges,
                                                               new_bdary_nodes )):
            bdary_edge1 = (bdary_mid_node, new_bdary_nodes[k+1])

            if _nodes_coincide( curve_edge[0], bdary_mid_node ):
                # Special case: start node of the curve is on the boundary.
                curve_edge0, curve_edge1 = \
                     _edge_pair_for_special_bdary_intersection( curve_edge, curve_list )
            else: # Start node of the curve edge is not on boundary.
                curve_id = curve_edge[0][0][0] # 1st part of id of the 1st node
                new_node_id = curve_new_node_id[ curve_id ]
                # Create a new node on the curve edge: ((curve_id,node_id), coord)
                curve_mid_node = ((curve_id, new_node_id), bdary_mid_node[1])
                curve_edge0 = (curve_edge[0], curve_mid_node)
                curve_edge1 = (curve_mid_node, curve_edge[1])
                curve_new_node_id[ curve_id ] = new_node_id + 1

            start_node_id = curve_edge[0][0]
            if not intersection_list_index.has_key( start_node_id ):
                # intersection_list_index does not have start_node_id,
                # i.e. a previous boundary intersection not registered.
                new_intersections.append( (bdary_edge1, curve_edge0) )
                new_intersections.append( (curve_edge1, bdary_edge0) )
                intersection_list_index[ start_node_id ] = list_index
            else: # The edge starting with start_node_id was has
                  # intersected another boundary edge already.
                intersection1 = [ (bdary_edge1, curve_edge0),
                                  (curve_edge1, bdary_edge0) ]
                intersection0_index = intersection_list_index[ start_node_id ]
                _update_multiple_boundary_intersections( intersection1,
                                                         new_intersections,
                                                         intersection0_index,
                                                         curve_new_node_id )
            list_index = list_index + 2
            bdary_start_node = bdary_mid_node
            bdary_edge0 = bdary_edge1

    return new_intersections


def chop_up_curves(curve_list, intersection_list):
    """Makes a list of curve segments by cutting curves from intersections.

    Given a list of curve objects and a list of intersections,
    this function goes through all intersection locations and creates
    a list of curve segments that represent pieces of the curve
    cut at the intersection locations.

    Parameters
    ----------
    curve_list : list of Curve objects.
    intersection_list : list of tuples
        A list of intersecting edge pairs. Each edge is a pair
        (start_node, end_node) and each node is a tuple of the form
        ((curve_id, node_id), node_coord), where node_coord is a NumPy
        array storing the node coordinate.

    Returns
    -------
    curve_segments : dict
        A dictionary of curve segments, where the key is the start node,
        a pair of (curve_id, node_id) and the value is another pair
        (end node, coord array).
        Similar to start node, end node is a pair (curve_id, node_id).
        The coord array is a NumPy array of shape (2,n_coords), which
        stores the coordinates of all the nodes from start to end node,
        excluding the end node.
    broken_curve_ids : set
        The id numbers of the curves that have been chopped into
        curve segments.
    """

    # The method for chopping up the curves to create the curve segments
    # works in three stages:
    #
    # 1) We go through all the intersecting edges and record all start
    #    nodes and end nodes (of intersecting edges) for each curve.
    #    These nodes tell us where the curves will be cut.
    #    Also we add each intersecting edge to curve segments, because
    #    each of these edges is also a segment to be reconnected.
    #
    # 2) Having the start node sets and end node sets for each curve,
    #    we now need to list the segment start nodes and segment end
    #    nodes (to initiate cutting). The end node of an intersecting
    #    edge is the start of a succeeding curve segment. Similarly start
    #    nodes of intersecting edges are end nodes of preceding curve
    #    segments. However we need take care not to include nodes that
    #    are simultaneously start nodes and end nodes of intersecting
    #    edges. They are not part of curve segments.
    #
    # 3) Having a list of sequence start and end nodes for each curve,
    #    we go through the list and extract subarrays of the curve's
    #    coordinates that correspond to the sequence [start,end) (exclude
    #    end node coordinate). We store these in a dictionary curve_segments,
    #    which enables us to access the end node, coordinate array of
    #    a curve segment by specifying its start node.

    curve_segments = {}
    broken_curve_ids = set()
    intersections_start_node_set = {}
    intersections_end_node_set = {}

    # The following loop over all intersections adds all intersecting
    # edges to curve_segments. It also collects the start and end nodes
    # of intersecting edges in intersections_start_node_set and
    # intersections_end_node_set respectively.

    for edge0, edge1 in intersection_list:

        # EDGE0 # Get curve, node, coord info of the first edge
        (curve_id, start_node_id), start_coord  = edge0[0]
        end_node_id = edge0[1][0][1]
        # Add the first edge to curve segment dictionary
        curve_segments[ (curve_id, start_node_id) ] \
                        = [ (curve_id, end_node_id), start_coord ]
        broken_curve_ids.add( curve_id )
        # Add the start and end nodes to the curve's start & end node sets
        node_set = intersections_start_node_set.setdefault( curve_id, set() )
        node_set.add( start_node_id )
        node_set = intersections_end_node_set.setdefault( curve_id, set() )
        node_set.add( end_node_id )

        # EDGE1 # Get curve, node, coord info of the second edge
        (curve_id, start_node_id), start_coord  = edge1[0]
        end_node_id = edge1[1][0][1]
        # Add the first edge to curve segment dictionary
        curve_segments[ (curve_id, start_node_id) ] \
                        = [ (curve_id, end_node_id), start_coord ]
        broken_curve_ids.add( curve_id )
        # Add the start and end nodes to the curve's start & end node sets
        node_set = intersections_start_node_set.setdefault( curve_id, set() )
        node_set.add( start_node_id )
        node_set = intersections_end_node_set.setdefault( curve_id, set() )
        node_set.add( end_node_id )


    # We loop over all intersection start node sets (for every curve id)
    # and use them to define the segment start and segment ends.

    segment_boundaries = {}
    for curve_id, start_node_set in intersections_start_node_set.iteritems():
        # If start_node_set and end node set share some nodes,
        # they should not be included segment start and end nodes,
        # because they cannot be used to compute curve segments.
        end_node_set = intersections_end_node_set[ curve_id ]
        shared_nodes = start_node_set.intersection( end_node_set )
        start_node_set -= shared_nodes
        end_node_set   -= shared_nodes

        if len(start_node_set) > 0 and len(end_node_set) > 0:
            segment_start_nodes = sorted( end_node_set )
            segment_end_nodes   = sorted( start_node_set )

            if segment_start_nodes[0] < segment_end_nodes[0]:
                node_pairs = zip( segment_start_nodes, segment_end_nodes )
            else:
                node_pairs = zip( segment_start_nodes[:-1], segment_end_nodes[1:] )
                node_pairs.append((segment_start_nodes[-1], segment_end_nodes[0]))

            segment_boundaries[ curve_id ] = node_pairs


    # Extract the coordinate subarrays for each each curve segment
    # and store them in the dictionary 'curve_segments'.

    for curve_id, boundary_nodes in segment_boundaries.iteritems():

        coords = curve_list[ curve_id ].coords()

        for start, end in boundary_nodes:
            seq_start = (curve_id, start)
            seq_end   = (curve_id, end)
            curve_segments[ seq_start ] = [ seq_end, coords[:,start:end] ]

        # If start > end, the last coordinate subarray wasn't extracted
        # correctly. The correct value is assigned below.
        if start > end:
            curve_segments[ seq_start ][1] = np.hstack(( coords[:,start:],
                                                         coords[:,:end] ))

    return (curve_segments, broken_curve_ids)


def reconnect_curve_segments(curve_segments, intersection_list, boundary=None):
    """Reconnects given curve segments and returns the new curve list.

    Given a collection of curve segments and a list of intersecting
    edges, first reconnects the intersecting edges by swapping their
    end nodes, then follows paths of curve segments using their start
    and end nodes and connects the ones whose start and end points
    coincide. The outcome is a set of closed curves.

    Parameters
    ----------
    curve_segments : A dictionary of curve segments, where the key
        is the start node, a pair of (curve_id, node_id) and the value
        is another pair (end node, coord array).
        Similar to start node, end node is a pair (curve_id, node_id).
        The coord array is a NumPy array of shape (2,n_coords), which
        stores the coordinates of all the nodes from start to end node,
        excluding the end node.
    intersection_list : list of tuples
        A list of intersecting edge pairs. Each edge is a pair
        (start_node, end_node) and each node is a tuple
        of the form ((curve_id, node_id), node_coord), where node_coord
        is a NumPy array storing the node coordinate.

    Returns
    -------
    new_curve_list : list of Curve objects
        Each of the Curve objects in this list have been obtained by
        joining the corresponding curve segments.
    """

    # Reconnect the intersections. Note the edge and segment structures:
    #    (curve_id,node_id) of start node = edge[0][0]
    #    (curve_id,node_id) of end node   = edge[1][0]
    #     end node of edge = curve_segments[ start node ][0]
    if boundary is not None: # namely processing boundary intersections
        # Set edge1.end = edge0.start
        for edge0, edge1 in intersection_list:
            curve_segments[ edge1[0][0] ][0] = edge0[0][0]

    else: # processing curve-curve intersections (not boundary)
        # Swap end nodes of intersecting edges: swap( edge0.end, edge1.end )
        for edge0, edge1 in intersection_list:
            curve_segments[ edge0[0][0] ][0] = edge1[1][0]
            curve_segments[ edge1[0][0] ][0] = edge0[1][0]

    # Follow curve segments from start to end to create the new curves.
    new_curve_list = []
    while len( curve_segments ) > 0:

        # Get a start node from curve segment for the new curve.
        key, value = curve_segments.popitem()
        curve_start = start = key
        end, coords = value
        curve_coord_list = [ coords ]

        # Now collect the coordinates of curve segments: s1->e1->s2->...eN->s1.
        while end != curve_start:
            start = end
            end, coords = curve_segments.pop( start )
            curve_coord_list.append( coords )

        coords = np.hstack( curve_coord_list )
        if coords.shape[1] > 2: # don't add curves of only two nodes
            curve = Curve( coords, adaptivity_parameters=None )
            if (boundary is None) or _curve_inside_boundary( curve, boundary ):
                new_curve_list.append( curve )

    return new_curve_list


def adjust_update(curves, update, intersection_list, intersecting_edges, edge_info):
    return (intersection_list, intersecting_edges, edge_info)


min_curve_area = 1e-4  # Remove the curves smaller than this.

def update_topology(curves, update, intersection_info=None):

    # Detect and compute the intersections between the edges of the curves.
    if intersection_info is None:
        intersection_list, intersecting_edges, edge_info \
                           = compute_intersections( curves )
    else:
        intersection_list, intersecting_edges, edge_info = intersection_info

    # If there are no intersections, return without further action.
    if len( intersection_list ) == 0:
        small_curves = [ curve_id
                         for curve_id, curve in enumerate(curves.curve_list())
                         if np.abs(curve.area()) < min_curve_area ]
        # Still check for small curves, remove if necessary.
        if len(small_curves) > 0:
            broken_curve_ids = set(small_curves)
            curves.add_remove_curves( broken_curve_ids, [] )
        return (curves, update)

    # Reduce the size of the update if it creates conflicts or too big
    # to resolve the new topology well.
    # intersection_list, intersecting_edges, edge_info = \
    #                    adjust_update( curves, update, intersection_list,
    #                                   intersecting_edges, edge_info )

    # Refine the intersection locations, to increase the resolution
    # at these locations and eliminate multiple intersections.
    intersection_list = refine_intersecting_edges( curves.curve_list(),
                                                   intersection_list,
                                                   intersecting_edges,
                                                   edge_info )

    # Reconnect the intersecting edges to obtain the new curves.
    curve_segments, broken_curve_ids \
                    = chop_up_curves( curves.curve_list(),
                                      intersection_list )

    new_curve_list = reconnect_curve_segments( curve_segments,
                                               intersection_list )

    # The reconnections at the previous step create some superfluous
    # curves in addition to the new curves. Keep only the correct new
    # curves and the old untouched curves.
    curves.add_remove_curves( broken_curve_ids, new_curve_list )


    # If there are small curves, remove them from curve family.
    small_curves = [ curve_id
                     for curve_id, curve in enumerate(curves.curve_list())
                     if np.abs(curve.area()) < min_curve_area ]
    if len(small_curves) > 0:
        broken_curve_ids = set(small_curves)
        curves.add_remove_curves( broken_curve_ids, [] )


    # Return the updated curve family and the adjusted update.
    return (curves, update)


def adjust_curves_for_boundary(curves, boundary):
    if boundary is None:  return curves
    # Always use positively-oriented boundary.
    boundary.set_orientation(1)

    intersections, outside_curve_ids = \
            compute_boundary_intersections( curves.curve_list(), boundary )

    if (len(intersections) == 0) and (len(outside_curve_ids) == 0):
        if curves.orientation() > 0:
            return curves
        else: # curves.orientation() < 0
            curves = curves.copy()
            topmost_curve_ids = set( curves.curve_hierarchy()[0] )
            curves.add_remove_curves( [], [boundary], topmost_curve_ids, 1 )
            return curves

    # If all the curves are now outside, return empty or just boundary.
    if len(outside_curve_ids) == len(curves.curve_list()):
        if curves.orientation() > 0:
            return CurveHierarchy( [] )
        else: # curves.orientation() < 0
            return CurveHierarchy( [boundary], adaptivity_parameters=None )

    # Work on a copy of the curve family
    curves = curves.copy()

    # If there are no boundary intersections, remove outside curves & return.
    if len(intersections) == 0:
        curves.add_remove_curves( outside_curve_ids, [] ) # No new curves yet.
        if curves.orientation() < 0: # Now need to add boundary curve as topmost.
            topmost_curve_ids = set( curves.curve_hierarchy()[0] )
            curves.add_remove_curves( [], [boundary], topmost_curve_ids )
        return curves

    # We need to account for intersections with the boundary.

    curve_list = curves.curve_list()
    curve_list.append( boundary )
    boundary_curve_id = len(curve_list) - 1

    intersections = refine_boundary_intersections( curve_list,
                                                   intersections )
    curve_segments, broken_curve_ids = chop_up_curves( curve_list,
                                                       intersections )
    new_curve_list = reconnect_curve_segments( curve_segments,
                                               intersections, boundary )
    broken_curve_ids.update( outside_curve_ids )

    # Boundary curve was not in the original curve family, so take it out.
    broken_curve_ids.remove( boundary_curve_id )
    curve_list.pop() # Remove boundary curve from curve list.

    # Old topmost curves might need their parent info updated.
    topmost_curve_ids = set( curves.curve_hierarchy()[0] )
    recompute_parents = set.difference( topmost_curve_ids, broken_curve_ids )

    # Finally update the curve family by removing broken curves and
    # adding the new curves (boundary and reconnected curves).
    curves.add_remove_curves( broken_curve_ids, new_curve_list, recompute_parents )
    return curves
