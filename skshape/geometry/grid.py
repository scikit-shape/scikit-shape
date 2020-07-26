import numpy as np
from numpy import pi
from ..numerics.fem import grid_fem as FEM
from ._mesh import Mesh

UNMARKED = 0
MARKED_INSIDE = 1
MARKED_OUTSIDE = -1

class Grid2d(Mesh):

    def __init__(self, resolution, origin=(0.0,0.0), size=(1.0,1.0)):
        super(Grid2d, self).__init__()
        self.FEM = FEM
        self.resolution = resolution
        self.origin = origin
        self.size = size
        self.increment = ( size[0]/(resolution[0]-1), size[1]/(resolution[1]-1) )

    def dim(self):
        return 2

    def dim_of_world(self):
        return 2

    def mark_inside_outside(self, curves):
        grid = np.zeros( self.resolution, dtype=int )
        self._mark_curve_neighborhood( curves, grid )
        self._fast_sweeping( grid )
        return grid


    # def _fast_sweeping(self, grid):
    #     pass
    #
    # TODO:
    # - ANGLES DIVIDING THE CELL INTO FOUR QUADRANTS SHOULD RECOMPUTE
    #   ANGLES AT EVERY CELL INSTANCE.
    # - Check for edges outside.
    # - Check for edges partially outside.
    # - Edge marking may miss grid points for diagonal edges spanning
    #   three cells.
    # - Need special code for the case when many consecutive small
    #   elements lie in the same grid cell; in this case, the corners
    #   of the grid cell should be marked based on the incoming and
    #   outgoing edges.

    def _mark_curve_neighborhood(self, curves, grid):

        topmost_curve_ids, parent, children = curves.curve_hierarchy()
        curve_list = curves.curve_list()

        for curve_id, curve in enumerate(curve_list):

            ambiguous_points = []
            coords = curve.coords()
            n = coords.shape[1]
            I = np.empty(n,dtype=int)
            J = np.empty(n,dtype=int)
            I[:] = np.floor( (coords[0,:] - self.origin[0]) / self.increment[0] )
            J[:] = np.floor( (coords[1,:] - self.origin[1]) / self.increment[1] )

            for k in range(n):
                x0,y0 = coords[:,k]
                x1,y1 = coords[:,(k+1)%n]
                x2,y2 = coords[:,(k+2)%n]
                i0,j0 = I[k], J[k]
                i1,j1 = I[(k+1)%n], J[(k+1)%n]
                self._mark_node_neighborhood( grid, i1,j1, x0,y0,x1,y1,x2,y2,
                                              ambiguous_points )
                self._mark_edge_neighborhood( grid, i0,j0,i1,j1,
                                              x0,y0,x1,y1, ambiguous_points )

            for pt,(i,j) in ambiguous_points:
                child_curves = [ curve_list[k] for k in children[curve_id] ]
                if self._pt_inside_curve( pt, curve, child_curves ):
                    grid[i,j] = MARKED_INSIDE
                else:
                    grid[i,j] = MARKED_OUTSIDE


    def _pt_inside_curve(self,pt,curve,children):
        if curve.orientation() == 1:
            inside = curve.contains(pt)
            if inside:
                for child in children:
                    if not child.contains(pt):
                        inside = False
                        break
        else: # curve.orientation() == -1
            inside = not curve.contains(pt)
            if inside:
                for child in children:
                    if child.contains(pt):
                        inside = False
                        break
        return inside

    def _assign_sign(self,grid,i,j,sign,ambiguous_points):
        if grid[i,j] == UNMARKED:
            grid[i,j] = sign
        elif grid[i,j] != sign:
            x,y = np.double(i), np.double(j)
            ambiguous_points.append( (np.array((x,y)),(i,j)) )


    def _mark_node_neighborhood(self, grid,i,j, x0,y0,x1,y1,x2,y2, ambiguous_points):

        max_i, max_j = grid.shape
        if (i < 0) or (j < 0):  return
        if (i >= max_i) or (j >= max_j):  return

        # A triplet of coordinates representing two consecutive edges are
        # used to assign signs to the nodes/corners of the cell enclosing
        # the middle point (x1,y1) in the given grid.
        # The following is an illustration of the two edges and the cell:
        #
        #      node1    node0                     indices of the cell nodes
        #        o--------o                          node0: (i+1,j+1)
        #        | x1,y1  |   x0,y0                  node1: ( i ,j+1)
        #        |   o----<----o  incoming edge      node2: ( i , j )
        #        |   |    |                          node3: (i+1, j )
        #        o---V----o
        #      node2 |  node3         In this example, the incoming edge is
        #            |                between nodes 3 and 0; therefore its
        #            o x2,y2          position is: edge0_pos = 3.
        #          outgoing edge      For the outgoing edge, edge2_pos = 2.


        # Compute displacement and angle of the edges w.r.t. middle point.
        dx0, dy0 = x0-x1, y0-y1
        dx2, dy2 = x2-x1, y2-y1
        angle0 = np.arctan2( dy0, dx0 )
        angle2 = np.arctan2( dy2, dx2 )

        # Use the edge angle to determine the position of the edge,
        # namely, the node after which the edge comes.
        # For example, in the illustration above, the incoming edge enters
        # the cell between nodes 3 and 0, with an angle between -pi/4
        # and pi/4; therefore, it has edge0_pos = 3.
        # Similarly, the outgoing edge has: edge2_pos = 2.

        if -0.75*pi < angle0 <= -0.25*pi:
            edge0_pos = 2
        elif -0.25*pi < angle0 <= 0.25*pi:
            edge0_pos = 3
        elif 0.25*pi < angle0 <= 0.75*pi:
            edge0_pos = 0
        else:
            edge0_pos = 1

        if -0.75*pi < angle2 <= -0.25*pi:
            edge2_pos = 2
        elif -0.25*pi < angle2 <= 0.25*pi:
            edge2_pos = 3
        elif 0.25*pi < angle2 <= 0.75*pi:
            edge2_pos = 0
        else:
            edge2_pos = 1

        # Assign the marks to the four corner nodes of the cell, based
        # on counterclockwise ordering. The nodes that fall in the range
        # starting from angle2/edge2_pos to angle0/edge0_pos are marked
        # "inside", the remaining nodes are marked "outside".
        # In the illustration above, node 3 is inside, nodes 0,1,2 are
        # outside.

        if edge0_pos == edge2_pos:
            if edge0_pos == 1:
                if angle0 < 0.0:  angle0 = angle0 + 2.0*pi
                if angle2 < 0.0:  angle2 = angle2 + 2.0*pi
            if angle2 > angle0:
                sign0 = sign1 = sign2 = sign3 = MARKED_INSIDE
            else:
                sign0 = sign1 = sign2 = sign3 = MARKED_OUTSIDE

        elif edge0_pos > edge2_pos:
            sign0 = MARKED_OUTSIDE

            if edge2_pos == 0:
                sign1 = MARKED_INSIDE
                if edge0_pos == 1:
                    sign2 = sign3 = MARKED_OUTSIDE
                elif edge0_pos == 2:
                    sign2 = MARKED_INSIDE
                    sign3 = MARKED_OUTSIDE
                else: # edge0_pos == 3
                    sign2 = sign3 = MARKED_INSIDE

            else: # edge2_pos == 1,2
                sign1 = MARKED_OUTSIDE
                if edge2_pos == 2: # and edge0_pos == 3 (b/c edge0pos>edge2pos)
                    sign2 = MARKED_OUTSIDE
                    sign3 = MARKED_INSIDE
                elif edge0_pos == 2: # and edge2_pos == 1
                    sign2 = MARKED_INSIDE
                    sign3 = MARKED_OUTSIDE
                else: # edge0_pos == 3 and edge2_pos == 1
                    sign2 = sign3 = MARKED_INSIDE

        else: # edge2_pos > edge0_pos
            sign0 = MARKED_INSIDE

            if edge0_pos == 0:
                sign1 = MARKED_OUTSIDE
                if edge2_pos == 1:
                    sign2 = sign3 = MARKED_INSIDE
                elif edge2_pos == 2:
                    sign2 = MARKED_OUTSIDE
                    sign3 = MARKED_INSIDE
                else: # edge2_pos == 3
                    sign2 = sign3 = MARKED_OUTSIDE

            else: # edge0_pos = 1,2
                sign1 = MARKED_INSIDE
                if edge0_pos == 2: # and edge2_pos == 3 (b/c edge2pos>edge0pos)
                    sign2 = MARKED_INSIDE
                    sign3 = MARKED_OUTSIDE
                elif edge2_pos == 2: # and edge0_pos == 1
                    sign2 = MARKED_OUTSIDE
                    sign3 = MARKED_INSIDE
                else: # edge2_pos == 3 and edge0_pos == 2
                    sign2 = sign3 = MARKED_OUTSIDE

        self._assign_sign( grid, i+1,j+1, sign0, ambiguous_points )
        self._assign_sign( grid,  i ,j+1, sign1, ambiguous_points )
        self._assign_sign( grid,  i , j , sign2, ambiguous_points )
        self._assign_sign( grid, i+1, j , sign3, ambiguous_points )


    def _mark_edge_neighborhood(self, grid, i0,j0,i1,j1,
                                x0,y0,x1,y1, ambiguous_points):
        max_i, max_j = grid.shape
        if (i0 < 0) and (i1 < 0):  return
        if (j0 < 0) and (j1 < 0):  return
        if (i0 >= max_i) and (i1 >= max_i):  return
        if (j0 >= max_j) and (j1 >= max_j):  return
        if (i0 == i1) and (np.abs(i0-i1) < 3):  return
        if (j0 == j1) and (np.abs(j0-j1) < 3):  return
        # TODO: The following check needs to be refined.
        # There is an unnecessary ambiguity.
        if (np.abs(i0-i1) < 1) and (np.abs(j0-j1) < 1):  return
        if i0 < 0:  i0 = 0
        if i1 < 0:  i1 = 0
        if j0 < 0:  j0 = 0
        if j1 < 0:  j1 = 0
        if i0 >= max_i:  i0 = max_i-1
        if i1 >= max_i:  i1 = max_i-1
        if j0 >= max_j:  j0 = max_j-1
        if j1 >= max_j:  j1 = max_j-1

        if i0 == i1: # Case of a vertical edge.
            i1 = i0 + 1

            if j0 < j1:
                sign0, sign1 = MARKED_INSIDE, MARKED_OUTSIDE
            else: # j0 > j1
                sign0, sign1 = MARKED_OUTSIDE, MARKED_INSIDE
                j0, j1 = j1, j0

            for j in range(j0+2,j1):
                self._assign_sign( grid, i0, j, sign0, ambiguous_points )
                self._assign_sign( grid, i1, j, sign1, ambiguous_points )

        elif j0 == j1: # Case of a horizontal edge.
            j1 = j0 + 1

            if i0 < i1:
                sign0, sign1 = MARKED_OUTSIDE, MARKED_INSIDE
            else: # i0 > i1
                sign0, sign1 = MARKED_INSIDE, MARKED_OUTSIDE
                i0, i1 = i1, i0

            for i in range(i0+2,i1):
                self._assign_sign( grid, i, j0, sign0, ambiguous_points )
                self._assign_sign( grid, i, j1, sign1, ambiguous_points )

        else: # Neither vertical nor horizontal edge.
            dx = x1 - x0
            dy = y1 - y0

            if np.abs(dx) > np.abs(dy):
                m = dy/dx
                if i0 < i1:
                    sign0, sign1 = MARKED_OUTSIDE, MARKED_INSIDE
                else: # i0 > i1
                    sign0, sign1 = MARKED_INSIDE, MARKED_OUTSIDE
                    i0, i1 = i1, i0

                offset = y0 - m*x0
                for i in range(i0+2,i1):
                    j0 = int(np.floor( m*i + offset ))
                    j1 = j0+1
                    self._assign_sign( grid, i, j0, sign0, ambiguous_points )
                    self._assign_sign( grid, i, j1, sign1, ambiguous_points )

            else: # np.abs(dx) <= np.abs(dy)
                m = dx/dy
                if j0 < j1:
                    sign0, sign1 = MARKED_INSIDE, MARKED_OUTSIDE
                else: # j0 > j1
                    sign0, sign1 = MARKED_OUTSIDE, MARKED_INSIDE
                    j0, j1 = j1, j0

                offset = x0 - m*y0
                for j in range(j0+2,j1):
                    i0 = int(np.floor( m*j + offset ))
                    i1 = i0+1
                    self._assign_sign( grid, i0, j, sign0, ambiguous_points )
                    self._assign_sign( grid, i1, j, sign1, ambiguous_points )


    ## def _mark_edge_neighborhood(self, grid, x0, y0, x1, y1, ambiguous_points):

    ##     if x0 == x1: # Case of a vertical edge.
    ##         i0 = int(np.floor(x0))
    ##         i1 = i0 + 1

    ##         if y0 < y1:
    ##             sign0, sign1 = MARKED_INSIDE, MARKED_OUTSIDE
    ##             j0 = int(np.ceil(y0)) + 1
    ##             j1 = int(np.floor(y1)) - 1
    ##         else: # y0 > y1
    ##             sign0, sign1 = MARKED_OUTSIDE, MARKED_INSIDE
    ##             j0 = int(np.ceil(y1)) + 1
    ##             j1 = int(np.floor(y0)) - 1

    ##         for j in range(j0,j1+1):
    ##             self._assign_sign( grid, i0, j, sign0, ambiguous_points )
    ##             self._assign_sign( grid, i1, j, sign1, ambiguous_points )

    ##     elif y0 == y1: # Case of a horizontal edge.
    ##         j0 = int(np.floor(y0))
    ##         j1 = j0 + 1

    ##         if x0 < x1:
    ##             sign0, sign1 = MARKED_OUTSIDE, MARKED_INSIDE
    ##             i0 = int(np.ceil(x0)) + 1
    ##             i1 = int(np.floor(x1)) - 1
    ##         else: # x0 > x1
    ##             sign0, sign1 = MARKED_INSIDE, MARKED_OUTSIDE
    ##             i0 = int(np.ceil(x1)) + 1
    ##             i1 = int(np.floor(x0)) - 1

    ##         for i in range(i0,i1+1):
    ##             self._assign_sign( grid, i, j0, sign0, ambiguous_points )
    ##             self._assign_sign( grid, i, j1, sign1, ambiguous_points )

    ##     else: # Neither vertical nor horizontal edge.
    ##         dx = x1 - x0
    ##         dy = y1 - y0

    ##         if np.abs(dx) > np.abs(dy):
    ##             m = dy/dx
    ##             if x0 < x1:
    ##                 sign0, sign1 = MARKED_OUTSIDE, MARKED_INSIDE
    ##                 i0 = int(np.ceil(x0)) + 1
    ##                 i1 = int(np.floor(x1)) - 1
    ##             else: # x0 > x1
    ##                 sign0, sign1 = MARKED_INSIDE, MARKED_OUTSIDE
    ##                 i0 = int(np.ceil(x1)) + 1
    ##                 i1 = int(np.floor(x0)) - 1

    ##             offset = y0 - m*x0
    ##             for i in range(i0,i1+1):
    ##                 j0 = int(np.floor( m*i + offset ))
    ##                 j1 = j0+1
    ##                 self._assign_sign( grid, i, j0, sign0, ambiguous_points )
    ##                 self._assign_sign( grid, i, j1, sign1, ambiguous_points )

    ##         else: # np.abs(dx) <= np.abs(dy)
    ##             m = dx/dy
    ##             if y0 < y1:
    ##                 sign0, sign1 = MARKED_INSIDE, MARKED_OUTSIDE
    ##                 j0 = int(np.ceil(y0)) + 1
    ##                 j1 = int(np.floor(y1)) - 1
    ##             else: # y0 > y1
    ##                 sign0, sign1 = MARKED_OUTSIDE, MARKED_INSIDE
    ##                 j0 = int(np.ceil(y1)) + 1
    ##                 j1 = int(np.floor(y0)) - 1

    ##             offset = x0 - m*y0
    ##             for j in range(j0,j1+1):
    ##                 i0 = int(np.floor( m*j + offset ))
    ##                 i1 = i0+1
    ##                 self._assign_sign( grid, i0, j, sign0, ambiguous_points )
    ##                 self._assign_sign( grid, i1, j, sign1, ambiguous_points )
