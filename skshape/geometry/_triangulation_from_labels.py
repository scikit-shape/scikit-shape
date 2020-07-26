"""Finds, returns, and plots the curves in a labeled pixel image.

Determines the curves by parsing the image provided cell by cell once.
Curves include the border curves, which have the highest ids. Curves end
at junctions, when they intersect boundary curves, or when they intersect
with themselves.

"""

import numpy as np
from meshpy import triangle
from collections import deque
from ..numerics.function import ImageFunction
from ..image.synthetic_image import BoxFunction
from scipy.ndimage import map_coordinates, spline_filter, convolve
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import minimize


def triangulation_from_labels(image, smooth_boundaries=True, coarsen_boundaries=True):
    """Creates a triangulation for the regions of the labeled image.

    This function identifies the regions in a given labeled image, extracts
    the boundary curves between regions, and creates a triangulation the conforms
    to the regions.

    Parameters
    ----------
    image : NumPy array
        An image of pixels labels
    smooth_boundaries : bool, optional
        Boolean flag that determines whether or not the region boundary
        curves will be smoothed. Default value is True.
    coarsen_boundaries : bool, optional
        Boolean flag that determines whether or not the region boundary
        curves will be coarsened. Default value is True.

    Returns
    -------
    boundary_curves : list of NumPy arrays
        List of region boundary curves stores in two-row arrays.
    tri_nodes : NumPy array
        (3,n)-sized array of floats, storing the coordinates of all
        the nodes in the triangulation.
    triangles : NumPy array
        (3,nT)-sized array of integers, storing the nodes of the triangles.

    """

    def count_labels(i, j, image):
        count = 1
        labels = set((image[i][j],))
        if not image[i][j + 1] in labels:
            labels.add(image[i][j + 1])
            count += 1
        if not image[i + 1][j + 1] in labels:
            labels.add(image[i + 1][j + 1])
            count += 1
        if not image[i + 1][j] in labels:
            count += 1
        return count

    def check_cell(i, j, image):
        cell_pts = []
        if image[i][j + 1] != image[i + 1][j + 1]: # Check right
            cell_pts.append((i + 0.5, j + 1))
        if image[i + 1][j + 1] != image[i + 1][j]: # Check up
            cell_pts.append((i + 1, j + 0.5))
        if image[i + 1][j] != image[i][j]: # Check left
            cell_pts.append((i + 0.5, j))
        if image[i][j] != image[i][j + 1]: # Check down
            cell_pts.append((i, j + 0.5))
        return cell_pts

    curves = {} # {curve_id: list containing 2 deques representing x and y pts}
    unconnected_pts = {} # Endpts of unfinished curves {(i,j): curve_id}
    next_curve_id = 0 # Next open keys for insertion of new curves
    junctions = {} # {junction pt: set of curve_ids}
    boundary_pts = set() # Set of boundary points

    # Traverses picture once to trace curves
    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1] - 1):
            num_labels = count_labels(i, j, image)
            pts_to_add = check_cell(i, j, image)
            if len(pts_to_add) == 4:
                if num_labels < 4: # Arbitrary diagonal case
                    next_curve_id = _append_segment(pts_to_add[0], pts_to_add[3],
                                                   unconnected_pts, curves,
                                                   next_curve_id, image.shape,
                                                   junctions, boundary_pts)
                    next_curve_id = _append_segment(pts_to_add[1], pts_to_add[2],
                                                   unconnected_pts, curves,
                                                   next_curve_id, image.shape,
                                                   junctions, boundary_pts)
                else: # 4-way junction
                    junctions[(i + 0.5, j + 0.5)] = set()
                    for current_pt in pts_to_add:
                        next_curve_id = _append_segment(current_pt,
                                                       (i + 0.5, j + 0.5),
                                                       unconnected_pts, curves,
                                                       next_curve_id,
                                                       image.shape, junctions,
                                                       boundary_pts)
            elif num_labels == 3: # 3-way junction
                junctions[(i + 0.5, j + 0.5)] = set()
                for current_pt in pts_to_add:
                    next_curve_id = _append_segment(current_pt,
                                                   (i + 0.5, j + 0.5),
                                                   unconnected_pts, curves,
                                                   next_curve_id, image.shape,
                                                   junctions, boundary_pts)
            elif num_labels == 2: # Non-junction contour
                next_curve_id = _append_segment(pts_to_add[0], pts_to_add[1],
                                               unconnected_pts, curves,
                                               next_curve_id, image.shape,
                                               junctions, boundary_pts)

    vtx, tri, curves = _get_curve_grain_output( curves, junctions, boundary_pts,
                                                image.shape, next_curve_id )

    if smooth_boundaries:
        curves = smooth_all_curves( curves, image, 1.0 )

    if coarsen_boundaries:
        curves = coarsen_all_curves( curves, 2e-3 )

    if smooth_boundaries or coarsen_boundaries:
        vtx,tri = triangulate( curves )

    return  (curves, vtx, tri)
    # return (curves, junctions, boundary_pts, next_curve_id)

def _scale_curves(curves, dim, boundary_curves = None):
    scaled_curves = []
    scaled_internal_curves = []
    scaled_boundary_curves = []
    h = 1.0 / (min(dim)-1)
    for key in curves.keys():
        scaled_curves.append(np.array((curves[key][0], curves[key][1])) * h)
        if boundary_curves is not None:
            if key in boundary_curves:
                scaled_boundary_curves.append(scaled_curves[-1])
            else:
                scaled_internal_curves.append(scaled_curves[-1])
    if boundary_curves is None:
        return scaled_curves
    else:
        return [scaled_curves, scaled_internal_curves, scaled_boundary_curves]

def _scale_pts(pt_set, dim):
    scaled_pts = set()
    h = 1.0 / (min(dim)-1)
    for pt in pt_set:
        scaled_pt = (pt[0] * h, pt[1] * h)
        scaled_pts.add(scaled_pt)
    return scaled_pts

def _feq(float1, float2, threshold=1e-6):
    return abs(float1 - float2) < threshold

def _get_curve_grain_output(curves, junctions, boundary_pts, dim, next_curve_id):
    # Find boundary curves and boundary curve ids
    boundary_info = _find_boundaries(curves, junctions, boundary_pts, dim, next_curve_id)
    boundary_curves = boundary_info[0]
    next_curve_id = boundary_info[1]

    # Find grains
    #grains = g.find_grains(copy.deepcopy(curves), boundary_curves, junctions, dim, next_curve_id)
    grains = None

    # Scale, plot and return list of curves
    categorized_curves = _scale_curves(curves, dim, boundary_curves)
    curves = categorized_curves[0]
    internal_curves = categorized_curves[1]
    boundary_curves = categorized_curves[2]

    #plot_curves(curves)


    # Compile the set of internal junctions (remove the boundary points from
    # the junctions set)
    scaled_dim = ( (dim[0]-1)/(min(dim)-1), (dim[1]-1)/(min(dim)-1) )
    junctions_set = set()
    junctions_set.update(junctions.keys())
    junctions_set = _scale_pts(junctions_set, dim)
    internal_junctions = set()
    for pt in junctions_set:
        if (not _feq(pt[0],0) and not _feq(pt[0], scaled_dim[0]) and not _feq(pt[1],0) and
            not _feq(pt[1], scaled_dim[1])):
            internal_junctions.add(pt)
    junctions = internal_junctions

    # Compile the set of boundary points (add the corners of the image too)
    boundary_pts = _scale_pts(boundary_pts, dim)
    boundary_pts.add((0,0))
    boundary_pts.add((0,scaled_dim[1]))
    boundary_pts.add((scaled_dim[0],0))
    boundary_pts.add((scaled_dim[0],scaled_dim[1]))

    factor = 1.0 / (min(dim) - 1)

    mesh = triangulate( curves )

    return (mesh[0], mesh[1], curves)
    # return [ curves, grains, internal_curves, boundary_curves, junctions,
    #          boundary_pts, mesh[0], mesh[1] ]

def triangulate(curves):

    # Collect the curves and segments
    pts = np.zeros((2,0)) # Amalgamates all points used in curves
    segments = set() # Amalgamates all segments in the the image
    used_pts = {} # {(x,y): number}
    for curve in curves:
        # Avoid duplicating points by checking whether endpoints have been used
        endpt_1 = (curve[0][0], curve[1][0])
        endpt_2 = (curve[0][-1], curve[1][-1])
        curve_len = len(curve[0])
        l = len(pts[0]) # Previous length of pts before this curve's points
        if endpt_1 in used_pts and endpt_2 in used_pts:
            pts = np.hstack([pts,curve[:,1:-1]])
            if curve_len == 2:
                segments.add((used_pts[endpt_1], used_pts[endpt_2]))
            elif curve_len == 3:
                segments.add((used_pts[endpt_1], l))
                segments.add((l, used_pts[endpt_2]))
            else:
                segments.update([(i + l - 1, i + l) for i in range(1, curve_len - 2)])
                segments.add((used_pts[endpt_1], l))
                segments.add((len(pts[0]) - 1, used_pts[endpt_2]))

        elif endpt_1 in used_pts:
            pts = np.hstack([pts,curve[:,1:]])
            segments.update([(i + l - 1, i + l) for i in range(1, curve_len - 1)])
            segments.add((used_pts[endpt_1], l))
            used_pts[endpt_2] = len(pts[0]) - 1

        elif endpt_2 in used_pts:
            pts = np.hstack([pts,curve[:,:-1]])
            segments.update([(i + l, i + l + 1) for i in range(curve_len - 2)])
            segments.add((len(pts[0]) - 1, used_pts[endpt_2]))
            used_pts[endpt_1] = l

        else:
            used_pts[endpt_1] = l
            if _feq(endpt_1[0], endpt_2[0]) and _feq(endpt_1[1], endpt_2[1]):
                pts = np.hstack([pts,curve[:,:-1]])
                segments.update([(i + l, i + l + 1) for i in range(curve_len - 2)])
                segments.add((len(pts[0]) - 1, used_pts[endpt_1]))
            else:
                pts = np.hstack([pts,curve])
                segments.update([(i + l, i + l + 1) for i in range(curve_len - 1)])
                used_pts[endpt_2] = len(pts[0]) - 1

    # Get mesh using triangle from MeshPy
    pts = pts.T
    segments = np.array( list( segments ) )

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points( pts )
    mesh_info.set_facets( segments )

    triangulation = triangle.build( mesh_info )

    mesh_triangles = np.array( triangulation.elements )
    mesh_pts = np.array( triangulation.points )

    return (mesh_pts, mesh_triangles)

def _append_segment(pt1, pt2, unconnected_pts, curves, next_curve_id, dim,
                    junctions, boundary_pts):
    """Appends segments of contour lines to the appropriate curves.

    Given two points that represent a segment to add to a contour curve,
    append_pt figures out to which existing curves the segment should be added
    (if any).  Starts a new curve in the dictionary curves if no match found.

    Parameters
    ----------
    pt1 : array_like
        First point in segment of curve
    pt2 : array_like
        Second point in segment of curve
    unconnected_pts : dict
        Dictionary with keys, which are endpoints of contour curves
        that are not yet finished, and values, which are keys of
        corresponding curves
    curves : dict
        Dictionary, with keys, which are arbitrary integers, and values,
        which are lists of lists representing points on contour curves
    next_curve_id : int
        The next available id for a new curve in the dictionary
    dim : tuple
        Dimensions of image (row, col)
    junctions : dict
        Dictionary of junction pts, the values are sets of connected
        curve ids
    boundary_pts : set
        Set of ids that represent boundary curve pieces

    Returns
    -------
    next_curve_id : int
        Integer representing a key available for the next new curve
    """

    def is_junction_pt(pt):
        return not _feq(pt[0], int(pt[0])) and not _feq(pt[1], int(pt[1]))

    def is_boundary_pt(pt, dim):
        if _feq(pt[0],0) or _feq(pt[0], dim[0] - 1):
            return True
        elif _feq(pt[1], 0) or _feq(pt[1], dim[1] - 1):
            return True
        return False

    # Adds junctions at boundary points
    if pt1 not in boundary_pts and is_boundary_pt(pt1, dim):
        junctions[pt1] = set()
        boundary_pts.add(pt1)
    if pt2 not in boundary_pts and is_boundary_pt(pt2, dim):
        junctions[pt2] = set()
        boundary_pts.add(pt2)

    # Case 1: Segment added will adjoin two curves
    if pt1 in unconnected_pts and pt2 in unconnected_pts:
        _adjoin_curves(pt1, pt2, unconnected_pts, curves, junctions, boundary_pts)

    # Cases 2 & 3: Segment should be appended to an existing curve
    elif bool(pt1 in unconnected_pts) ^ bool(pt2 in unconnected_pts):
        end_pt = None
        new_pt = None
        curve_id = None
        # Case 2
        if pt1 in unconnected_pts:
            end_pt = pt1
            new_pt = pt2
            curve_id = unconnected_pts[pt1]
        # Case 3
        else:
            end_pt = pt2
            new_pt = pt1
            curve_id = unconnected_pts[pt2]

        # Updates junctions if pt1 is on boundary
        if pt1 in junctions:
            junctions[pt1].add(curve_id)
        # Updates junctions if pt2 is on boundary or is a junction
        elif pt2 in junctions:
            junctions[pt2].add(curve_id)

        # Checks which end of the curve the point should be added and checks
        # for extra points on the curve
        if (_feq(end_pt[0], curves[curve_id][0][0]) and
            _feq(end_pt[1], curves[curve_id][1][0])):
            if not _replace_pt(pt1, pt2, curves, curve_id, 1, dim):
                curves[curve_id][0].appendleft(new_pt[0])
                curves[curve_id][1].appendleft(new_pt[1])
        else:
            if not _replace_pt(pt1, pt2, curves, curve_id, -2, dim):
                curves[curve_id][0].append(new_pt[0])
                curves[curve_id][1].append(new_pt[1])

        # Adjust endpoint in unconnected_pts if necessary
        if not is_boundary_pt(new_pt, dim) and not is_junction_pt(new_pt):
            unconnected_pts[new_pt] = unconnected_pts[end_pt]
        del unconnected_pts[end_pt]

    # Case 4: Segment is not connected to any other curve
    else:
        curves[next_curve_id] = [ deque((pt1[0], pt2[0])),
                                  deque((pt1[1], pt2[1])) ]
        # Check that point is not a boundary or junction before insertion
        if not is_boundary_pt(pt1, dim) and not is_junction_pt(pt1):
            unconnected_pts[pt1] = next_curve_id
        else:
            junctions[pt1].add(next_curve_id)

        if not is_boundary_pt(pt2, dim) and not is_junction_pt(pt2):
            unconnected_pts[pt2] = next_curve_id
        else:
            junctions[pt2].add(next_curve_id)

        next_curve_id += 1 # Change next availabe curve_id
    return next_curve_id

def _adjoin_curves(pt1, pt2, unconnected_pts, curves, junctions, boundary_pts):
    """Finishes closed curves or adjoins two different curves.

    Given two points, both of which are connected to curve[s], this
    function decides how to incorporate the segment represented by pt1
    and pt2. If both curves are connected to the same curve, then the
    curve is a closed curve. If not, the two different curves are added
    together, keeping in mind the orientations of the curves.

    Parameters
    ----------
    pt1 : array_like
        First point in segment of curve
    pt2 : array_like
        Second point in segment of curve
    unconnected_pts : dict
        Dictionary with keys, which are endpoints of contour curves
        that are not yet finished, and values, which are keys of
        corresponding curves
    curves : dict
        Dictionary with keys, which are arbitrary integers, and values,
        which are lists of lists that represents points on contour curves.
    junctions : dict
        Dictionary of junctions pt keys and set values of connected
        curve ids
    boundary_pts : set
        Set of ids that represents boundary curve pieces

    Returns
    -------
    None

    """

    curve1_id = unconnected_pts[pt1]
    curve2_id = unconnected_pts[pt2]

    if curve1_id == curve2_id: # Closed curve, close the loop
        # pt1 is at beginning of curve
        if (_feq(pt1[0], curves[curve1_id][0][0]) and
            _feq(pt1[1], curves[curve1_id][1][0])):
            _rm_adjoining_pts(pt1, pt2, curve1_id, 1, curve1_id, -2, curves)
        else:
            _rm_adjoining_pts(pt1, pt2, curve1_id, -2, curve1_id, 1, curves)
        curves[curve1_id][0].append(curves[curve1_id][0][0])
        curves[curve1_id][1].append(curves[curve1_id][1][0])

    else: # Two different curves must be adjoined
        if (_feq(pt1[0], curves[curve1_id][0][0]) and
            _feq(pt1[1], curves[curve1_id][1][0])):
            # Both points at beginning
            # Reverse curve 1 and append to beginning of curve 2
            if (_feq(pt2[0], curves[curve2_id][0][0]) and
                _feq(pt2[1], curves[curve2_id][1][0])):
                _rm_adjoining_pts(pt1, pt2, curve1_id, 1, curve2_id, 1, curves)
                curves[curve2_id][0].extendleft(curves[curve1_id][0])
                curves[curve2_id][1].extendleft(curves[curve1_id][1])

            # Points are on opposite ends, add curve 1 to the end of curve 2
            else:
                _rm_adjoining_pts(pt1, pt2, curve1_id, 1, curve2_id, -2, curves)
                curves[curve2_id][0].extend(curves[curve1_id][0])
                curves[curve2_id][1].extend(curves[curve1_id][1])

            # Delete curve 1 and change id in unconnected_pts if necessary
            c1_last_pt = (curves[curve1_id][0][-1], curves[curve1_id][1][-1])
            if c1_last_pt in unconnected_pts:
                unconnected_pts[c1_last_pt] = curve2_id
            if c1_last_pt in junctions:
                       junctions[c1_last_pt].remove(curve1_id)
                       junctions[c1_last_pt].add(curve2_id)
            del curves[curve1_id]

        else:
            # Both points are at end, curve 2 needs to be reversed
            if (_feq(pt2[0], curves[curve2_id][0][-1]) and
                _feq(pt2[1], curves[curve2_id][1][-1])):
                curves[curve2_id][0] = deque(reversed(curves[curve2_id][0]))
                curves[curve2_id][1] = deque(reversed(curves[curve2_id][1]))
            # Add curve 2 to end of curve 1
            _rm_adjoining_pts(pt1, pt2, curve1_id, -2, curve2_id, 1, curves)
            curves[curve1_id][0].extend(curves[curve2_id][0])
            curves[curve1_id][1].extend(curves[curve2_id][1])

            # Delete curve 2 and change id in unconnected_pts if necessary
            c2_last_pt = (curves[curve2_id][0][-1], curves[curve2_id][1][-1])
            if ((curves[curve2_id][0][-1], curves[curve2_id][1][-1]) in
                unconnected_pts):
                unconnected_pts[c2_last_pt] = curve1_id
            # Updates junctions for changed curve_id
            if c2_last_pt in junctions:
                junctions[c2_last_pt].remove(curve2_id)
                junctions[c2_last_pt].add(curve1_id)
            del curves[curve2_id]
    del unconnected_pts[pt1]
    del unconnected_pts[pt2]

def _find_boundaries(curves, junctions, boundary_pts, dim, next_curve_id):
    """Adds the boundary curves to the dictionary of curves.

    Determines the boundary curve pieces in an image.  Iterates through the
    boundary points from the top left corner (image index 0, 0), clockwise.
    When the pixel label changes, a new boundary curve is added.

    Parameters
    ----------
    curves : dict
        Dictionary with keys, which are arbitrary integers, and values,
        which are lists of lists that represents points on contour curves
    junctions : dict
        Dictionary of junctions pt keys and set values of connected
        curve ids
    boundary_pts : set
        Set of ids that represents boundary curve pieces
    next_curve_id : int
        The next available id for a new curve in the dictionary
    dim : tuple
        Tuple representing dimensions of image (row, col)

    Returns
    -------
    boundary_curve_ids: set
        Set of curve_ids for boundary curves

    """

    init_id = next_curve_id
    current_curve = [ deque((0,)), deque((0,)) ]
    boundary_curves = set() # set of curve_ids for boundary curves
    # Top edge
    for j in range(dim[1] - 1):
        if (0, j + 0.5) in boundary_pts:
            current_curve = _add_boundary_curve(curves, current_curve, junctions,
                                        (0, j + 0.5), next_curve_id,
                                        boundary_curves)
            next_curve_id += 1
        if j == dim[1] - 2: # Add top right corner
            current_curve[0].append(0)
            current_curve[1].append(j + 1)

    # Right edge
    for i in range(dim[0] - 1):
        if (i + 0.5, dim[1] - 1) in boundary_pts:
            current_curve = _add_boundary_curve(curves, current_curve, junctions,
                                        (i + 0.5, dim[1] - 1),
                                        next_curve_id, boundary_curves)
            next_curve_id += 1
        if i == dim[0] - 2: # Add bottom right corner
            current_curve[0].append(i + 1)
            current_curve[1].append(dim[1] - 1)

    # Bottom edge
    for j in range(dim[1] - 1, 0, -1):
        if (dim[0] - 1, j - 0.5) in boundary_pts:
            current_curve = _add_boundary_curve(curves, current_curve, junctions,
                                        (dim[0] - 1, j - 0.5),
                                        next_curve_id, boundary_curves)
            next_curve_id += 1
        if j == 1: # Add bottom left corner
            current_curve[0].append(dim[0] - 1)
            current_curve[1].append(0)

    # Left edge
    for i in range(dim[0] - 1, 0, -1):
        if (i - 0.5, 0) in boundary_pts:
            current_curve = _add_boundary_curve(curves, current_curve, junctions,
                                        (i - 0.5, 0), next_curve_id,
                                        boundary_curves)
            next_curve_id += 1
        if i == 1 and init_id == next_curve_id: # Add top left corner
            current_curve[0].append(0)
            current_curve[1].append(0)

    # If the border is never touched, the current curve is added to curves
    if next_curve_id == init_id:
        curves[init_id] = current_curve
        boundary_curves.add(init_id)
    # Handles connection at top left corner between piece on top edge and piece
    # on left edge
    else:
        # Adds the initial boundary curve to the end of the last boundary curve
        curves[next_curve_id] = current_curve
        curves[next_curve_id][0].extend(curves[init_id][0])
        curves[next_curve_id][1].extend(curves[init_id][1])
        # Update curve_id in other boundary_curves and junctions
        boundary_curves.remove(init_id)
        boundary_curves.add(next_curve_id)
        junction_index = (curves[init_id][0][-1], curves[init_id][1][-1])
        junctions[junction_index].remove(init_id)
        junctions[junction_index].add(next_curve_id)
        del curves[init_id]

    return [boundary_curves, next_curve_id]


def _add_boundary_curve(curves, current_curve, junctions, pt, next_curve_id,
                       boundary_curves):
    """Adds a boundary curve to the list of boundary curves.

    """

    # Adds the final point to the curve and then adds the curve to the dictionary
    current_curve[0].append(pt[0])
    current_curve[1].append(pt[1])
    curves[next_curve_id] = current_curve
    # Records the junction between the boundary curves
    junctions[pt].add(next_curve_id)
    junctions[pt].add(next_curve_id + 1)
    boundary_curves.add(next_curve_id)
    return [ deque((pt[0],)), deque((pt[1],)) ] # Starts a new curve

def _replace_pt(pt1, pt2, curves, curve_id, pt_index, dim):
    """Removes all but the endpoints of any straight line when extending curves.

    Checks for horizontal, vertical, and diagonal straight lines created by the
    addition of a new point.  If there are any extraneous points, the
    intermediate point is replaced with the new endpoint in the curve dictionary.

    Parameters
    ----------
    pt1 : array_like
        First point in segment of curve
    pt2 : array_like
        Second point in segment of curve
    unconnected_pts : dict
        Dictionary with keys, which are endpoints of contour curves
        that are not yet finished, and values, which are keys of
        corresponding curves
    curves : dict
        Dictionary with keys, which are arbitrary integers, and values,
        which are lists of lists that represents points on contour curves
    dim : tuple
        Dimensions of image (row, col)

    Returns
    -------
    A boolean representing whether a point was replaced or not.

    """

    def change_pt(repl_pt, pt_index, curves, curve_id):
        if pt_index == 1:
            curves[curve_id][0][0] = repl_pt[0]
            curves[curve_id][1][0] = repl_pt[1]
        else:
            curves[curve_id][0][-1] = repl_pt[0]
            curves[curve_id][1][-1] = repl_pt[1]

    replaced = False
    # Horizontal segment
    if _feq(pt1[0], pt2[0]) and _feq(curves[curve_id][0][pt_index], pt2[0]):
        if not _feq(pt2[1], int(pt2[1])):
            change_pt(pt2, pt_index, curves, curve_id)
        else:
            change_pt(pt1, pt_index, curves, curve_id)
        replaced = True

    # Vertical segment
    elif _feq(pt1[1], pt2[1]) and _feq(curves[curve_id][1][pt_index], pt2[1]):
        if not _feq(pt2[0], int(pt2[0])):
            change_pt(pt2, pt_index, curves, curve_id)
        else:
            change_pt(pt1, pt_index, curves, curve_id)
        replaced = True

    # Slanted segment in upper right or lower left corner
    elif _feq(pt2[0], pt1[0] - 0.5) and _feq(pt2[1], pt1[1] - 0.5):
        if _feq(curves[curve_id][1][pt_index],
               pt2[1] - (pt2[0] - curves[curve_id][0][pt_index])):
            change_pt(pt1, pt_index, curves, curve_id)
            replaced = True

    # Slanted segment in upper left corner.
    # Does not connect 2 curves since this function isn't called by adjoin_curves
    elif (_feq(pt1[1], int(pt1[1])) and _feq(pt2[0], pt1[0] - 0.5) and
          _feq(pt2[1], pt1[1]+ 0.5)):
        # Determine which pt is connected to the curve given
        pt1_curve = None
        if pt_index == 1:
            pt1_curve = (_feq(curves[curve_id][0][0], pt1[0]) and
                         _feq(curves[curve_id][1][0], pt1[1]))
        else:
            pt1_curve = (_feq(curves[curve_id][0][-1], pt1[0]) and
                         _feq(curves[curve_id][1][-1], pt1[1]))

        if pt1_curve:
            if (_feq(curves[curve_id][0][pt_index], pt1[0] + 0.5) and
                _feq(curves[curve_id][1][pt_index], pt1[1] - 0.5)):
                change_pt(pt2, pt_index, curves, curve_id)
                replaced = True
        else:
            if _feq(curves[curve_id][1][pt_index], pt2[1] +
                   (pt2[0] - curves[curve_id][0][pt_index])):
                change_pt(pt1, pt_index, curves, curve_id)
                replaced = True
    return replaced

def _rm_adjoining_pts(pt1, pt2, curve1_id, pt1_index, curve2_id, pt2_index,
                     curves):
    """Removes extraneous points when connecting two curves.

    """

    # Checks if previous point on curve 2 renders pt2 extraneous
    if _feq(curves[curve2_id][1][pt2_index], pt2[1] +
        (pt2[0] - curves[curve2_id][0][pt2_index])):

        if pt2_index == 1:
            curves[curve2_id][0].popleft()
            curves[curve2_id][1].popleft()
        else:
            curves[curve2_id][0].pop()
            curves[curve2_id][1].pop()

    # Checks if previous point on curve 1 renders pt1 extraneous
    if (_feq(pt1[0], curves[curve1_id][0][pt1_index] - 0.5) and
        _feq(pt1[1], curves[curve1_id][1][pt1_index] + 0.5)):

        if pt1_index == 1:
            curves[curve1_id][0].popleft()
            curves[curve1_id][1].popleft()
        else:
            curves[curve1_id][0].pop()
            curves[curve1_id][1].pop()


############################################################################


class _BoxIndicatorFunction(object):

    def __init__(self, min_x, max_x, transition_width):
        super( _BoxIndicatorFunction, self ).__init__()
        self._min_x = Xm = np.array( min_x )
        self._max_x = XM = np.array( max_x )
        self._eps = eps = transition_width
        self._f = BoxFunction( Xm - 0.5*eps, XM + 0.5*eps, eps )

    def _outside_pts(self, x):
        outside = np.zeros( x.shape[1], dtype=bool ) # No pt is outside yet
        for xi, min_xi, max_xi in zip( x, self._min_x, self._max_x):
            outside |= (xi < min_xi) | (max_xi < xi)
        outside = np.nonzero( outside )[0]
        return outside

    def __call__(self, x):
        y = np.ones( x.shape[1] )
        k = self._outside_pts(x)  # indices of outside points
        y[k] = self._f( x[:,k] )
        return y

    def gradient(self, x):
        grad = np.zeros_like(x)
        k = self._outside_pts(x)  # indices of outside points
        grad[:,k] = self._f.gradient( x[:,k] )
        return grad

    def hessian(self, x):
        dim = x.shape[1]
        hess = np.zeros((dim, dim, n_pts))
        k = self._outside_pts(x)  # indices of outside points
        hess[:,:,k] = self._f.hessian( x[:,k] )
        return hess

class _LabelEdgeIndicatorFunction(object):

    def __init__(self, image, sigma=2.0 ):
        self._I = I = image
        self.sigma = sigma

        edge_map = np.ones(I.shape, dtype=bool)

        test_neighbors = (I[1:,:] == I[:-1,:])
        edge_map[1:,:]  &= test_neighbors
        edge_map[:-1,:] &= test_neighbors

        test_neighbors = (I[:,1:] == I[:,:-1])
        edge_map[:,1:]  &= test_neighbors
        edge_map[:,:-1] &= test_neighbors

        test_neighbors = (I[1:,1:] == I[:-1,:-1])
        edge_map[1:,1:]   &= test_neighbors
        edge_map[:-1,:-1] &= test_neighbors

        test_neighbors = (I[1:,:-1] == I[:-1,1:])
        edge_map[1:,:-1] &= test_neighbors
        edge_map[:-1,1:] &= test_neighbors

        self._edge_map = ~edge_map

        G = np.double( edge_map )
        if (sigma is not None) and (sigma > 0.0):
            G = gaussian_filter( G, sigma )
            G -= G.min()
            G /= G.max()
        self._G = G

        self._g_fct = ImageFunction(G,3,2)

        n = min( image.shape )
        transition = factor = 1.0 / (n-1)

        min_x = (-0.5*transition, -0.5*transition )
        max_x = ( factor*image.shape[0] + 0.5*transition,
                  factor*image.shape[1] + 0.5*transition )

        self._indicator_fct = _BoxIndicatorFunction( min_x, max_x, transition )

    def __call__(self, x):
        G = self._g_fct(x)
        I = self._indicator_fct(x)
        return (G * I + (1.0 - I))

    def gradient(self, x):
        G = self._g_fct(x)
        I = (1.0 - self._indicator_fct(x))
        dG = self._g_fct.gradient(x)
        dI = -self._indicator_fct.gradient(x)
        return (I * dG + G * dI - dI)

    def hessian(self, x):
        raise NotImplementedError


class _WeightedCurveLength(object):

    def __init__(self, weight_fct, start_pt=None, end_pt=None):
        self._g = weight_fct
        self._X0 = start_pt
        self._Xn = end_pt

    def __call__(self, curve):
        X = curve
        open_curve = (self._X0 is not None) and (self._Xn is not None)
        if open_curve:
            X = np.vstack(( np.hstack([ self._X0[0], X[0], self._Xn[0] ]),
                            np.hstack([ self._X0[1], X[1], self._Xn[1] ]) ))
        else: # closed_curve:
            X = np.vstack(( np.hstack([ X[0], X[0,0] ]),
                            np.hstack([ X[1], X[1,0] ]) ))
        g = self._g(X)
        T = X[:,1:] - X[:,:-1]
        d = np.sqrt( T[0]**2 + T[1]**2 )
        avg_g = 0.5 * (g[1:] + g[:-1])
        return np.sum( d * avg_g )

    def gradient(self, curve):
        X = curve
        open_curve = (self._X0 is not None) and (self._Xn is not None)
        if open_curve:
            X = np.vstack(( np.hstack([ self._X0[0], X[0], self._Xn[0] ]),
                            np.hstack([ self._X0[1], X[1], self._Xn[1] ]) ))
        else: # closed_curve:
            X = np.vstack(( np.hstack([ X[0], X[0,0] ]),
                            np.hstack([ X[1], X[1,0] ]) ))
        g = self._g(X)
        dg = self._g.gradient(X)
        avg_g = 0.5 * (g[1:] + g[:-1])

        T = X[:,1:] - X[:,:-1]
        d = np.sqrt( T[0]**2 + T[1]**2 )
        T /= d

        if open_curve:
            avg_d = 0.5 * (d[1:] + d[:-1])
            grad = dg[:,1:-1] * avg_d + \
                   ( T[:,:-1] * avg_g[:-1] - T[:,1:] * avg_g[1:] )
        else: # closed curve
            avg_d = 0.5 * d
            avg_d[1:] += 0.5 * d[:-1]
            avg_d[0]  += 0.5 * d[-1]
            grad = dg * avg_d - avg_g * T
            grad[:,1:] += avg_g[:,:-1] * T[:,:-1]
            grad[:,0]  += avg_g[:,-1] * T[:,:-1]

        # Compute the projection of the gradient in the normal direction.
        curve_normals = _compute_normals(X)
        normal_grad = (grad * curve_normals).sum(0)
        projected_grad = normal_grad * curve_normals

        return projected_grad

    def hessian(self, curve):
        raise NotImplementedError


def _is_open_curve(curve):
    return ( np.linalg.norm(curve[:,0] - curve[:,-1]) > 10^-14 )


def smooth_all_curves(curves, image, sigma=1.0):
    weight_fct = _LabelEdgeIndicatorFunction( image, sigma )
    new_curves = []
    for old_curve in curves:
        if old_curve.shape[1] < 4:
            new_curves.append( old_curve.copy() )
        else:
            if _is_open_curve( old_curve ):
                start_node, end_node = old_curve[:,0], old_curve[:,-1]
            else:
                start_node, end_node = None, None
            energy = _WeightedCurveLength( weight_fct, start_node, end_node )
            new_curves.append( _smooth_curve( old_curve, energy ) )
    return new_curves


def _smooth_curve(curve, energy):
    options = {'gtol':1e-6,'disp':False}

    n = curve.shape[1]
    m = n-2 if _is_open_curve(curve) else n-1

    opt_func = lambda x: energy( x.reshape((2,m)) )
    opt_func_grad = lambda x: energy.gradient( x.reshape((2,m)) ).reshape((2*m) )

    x0 = curve[:,1:-1] if _is_open_curve( curve ) else curve[:,0:-1]
    x0 = np.reshape( x0, (2*m) )

    res = minimize(opt_func, x0, method='BFGS', jac=opt_func_grad, options=options)

    X = res.x.reshape((2,m))

    if _is_open_curve( curve ): # if the original curve was an open curve.
        new_curve = np.vstack(( np.hstack([ curve[0,0], X[0], curve[0,-1] ]),
                                np.hstack([ curve[1,0], X[1], curve[1,-1] ]) ))
    else: # closed curve
        new_curve = np.vstack(( np.hstack([ curve[0,0], X[0] ]),
                                np.hstack([ curve[1,0], X[1] ]) ))
    return new_curve

def _compute_normals(curve):
    # First compute element tangents and normals.
    T = curve[:,1:] - curve[:,:-1]
    N = np.vstack(( T[1], -T[0] ))
    # Compute node normals as weighted sums of the two neighboring elements.
    node_N = N[:,:-1] + N[:,1:]

    if not _is_open_curve( curve ):
        node_N0 = np.reshape( N[:,0]+N[:,-1], (2,1) )
        node_N  = np.hstack(( node_N0, node_N ))

    # Divide by vector length to get unit normals.
    node_N /= np.sqrt( node_N[0]**2 + node_N[1]**2)

    return node_N

def _compute_curvature(curve):
    x,y = curve

    dy1 = y[1:] - y[:-1]
    dy2 = y[2:] - y[:-2]

    D1 = dy1**2 + (x[1:] - x[:-1])**2
    D2 = dy2**2 + (x[2:] - x[:-2])**2

    bottom_sqr = D1[:-1] * D2 * D1[1:]

    K = x[:-2] * dy1[1:] - x[1:-1] * dy2 + x[2:] * dy1[:-1]
    K = -2.0 * K / np.sqrt( bottom_sqr )

    if not _is_open_curve( curve ):
        n = len(x) - 2
        d_0n = (x[0] - x[n])**2 + (y[0] - y[n])**2
        d_10 = (x[1] - x[0])**2 + (y[1] - y[0])**2
        d_1n = (x[1] - x[n])**2 + (y[1] - y[n])**2
        K0 = x[n]*(y[1] - y[0]) - x[0]*(y[1] - y[n]) + x[1]*(y[0] - y[n])
        K0 = -2.0 * K0 / np.sqrt( d_0n * d_10 * d_1n )
        K = np.hstack([ K0, K ])

    return K

def _compute_geometric_error(curve):
    n = curve.shape[1]
    K = np.abs( _compute_curvature( curve ) )
    elem_sizes = np.sum( (curve[:,1:] - curve[:,:-1])**2, 0 )**0.5

    elem_error = np.empty(n-1)
    elem_error[0] = K[0]
    elem_error[1:-1] = np.maximum( K[1:], K[:-1] )
    elem_error[-1] = K[-1]
    elem_error *= elem_sizes**2

    node_error = np.empty(n)
    node_error[0] = node_error[-1] = np.inf  # 1st & last won't be coarsened.
    node_error[1:-1] = np.maximum( elem_error[1:], elem_error[:-1] )
    return node_error

def _coarsen_curve(curve, tol=2e-3):
    if not _is_open_curve(curve):
        raise NotImplementedError

    error = _compute_geometric_error( curve )

    marked = error < tol
    marked_even = 2 * np.nonzero( marked[::2] )[0]
    marked[ marked_even + 1 ] = False

    new_curve = curve[:,~marked]
    return new_curve

def coarsen_all_curves(curves, tol=2e-3):
    max_iters = 10
    k = 0
    curves_changing = True
    while curves_changing and k < max_iters:
        new_curves = []
        curves_changing = False
        for old_curve in curves:
            if old_curve.shape[1] < 3:
                new_curves.append( old_curve.copy() )
            else:
                new_curve = _coarsen_curve( old_curve, tol )
                new_curves.append( new_curve )
                if new_curve.shape[1] < old_curve.shape[1]:
                    curves_changing = True
        curves = new_curves
        k = k+1
    return new_curves
