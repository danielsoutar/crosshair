import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def get_bounding_box(points): 
    """
    Read in a numpy array of shape M x 3, where M denotes
    the number of points. First column is xs, second ys, third zs.
    """
    x_min = np.min(points[:,0])
    x_max = np.max(points[:,0])
    y_min = np.min(points[:,1])
    y_max = np.max(points[:,1])
    z_min = np.min(points[:,2])
    z_max = np.max(points[:,2])

    max_range = np.array(
        [x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0

    mid_x = (x_max+x_min) * 0.5
    mid_y = (y_max+y_min) * 0.5
    mid_z = (z_max+z_min) * 0.5

    return [
        [mid_x - max_range, mid_x + max_range],
        [mid_y - max_range, mid_y + max_range],
        [mid_z - max_range, mid_z + max_range]
    ]


class Cube:
    """
    Wrapper around Poly3DCollection used for plotting in Matplotlib,
    as well as containing the actual points.
    """
    
    def __init__(self, points):
        self._faces = [
            (points[0], points[1], points[2], points[3]),  # bottom
            (points[0], points[4], points[7], points[3]),  # front face
            (points[0], points[1], points[5], points[4]),  # left face
            (points[3], points[7], points[6], points[2]),  # right face
            (points[1], points[5], points[6], points[2]),  # back face
            (points[4], points[5], points[6], points[7]),  # top
        ]

        self._points = np.array([p for p in points]).reshape((len(points), 3))
        self._polycollection = Poly3DCollection(self._faces)

    def get_polycollection(self):
        return self._polycollection

    def get_points(self):
        return self._points

    def get_faces(self):
        return self._faces


def construct_cube(base, length_vec, width_vec, height_vec):
    """
    Represent cube with 8 points, transform points into faces,
    and return a collection of these faces representing the cube.
    """
    point0 = base
    point1 = base + length_vec
    point2 = base + length_vec + width_vec
    point3 = base + width_vec
    point4 = base + height_vec
    point5 = base + length_vec + height_vec
    point6 = base + length_vec + width_vec + height_vec
    point7 = base + width_vec + height_vec

    return Cube([point0, point1, point2, point3,
                 point4, point5, point6, point7])


def format_cube(cube, facecolours=None, linewidths=None, edgecolours=None, alpha=None):
    """
    Make the cube pretty and the tings.
    """
    polycollection = cube.get_polycollection()

    polycollection.set_facecolor(facecolours)
    polycollection.set_linewidths(linewidths)
    polycollection.set_edgecolor(edgecolours)
    polycollection.set_alpha(alpha)

    return cube


def add_cube_to_space(space, cube):
    """
    Take in an axes-object as a 3D space, and add the cube to it. Make sure the space is
    large enough to encompass all objects in it. Return the space.
    """
    bounding_box = get_bounding_box(cube.get_points())

    if space["dims"] == [None, None, None]:
        dim = [bounding_box[0], bounding_box[1], bounding_box[2]]
    else:
        dim = [[min(space["dims"][i][0], bounding_box[i][0]), 
                max(space["dims"][i][1], bounding_box[i][1])] for i in range(len(bounding_box))]

    space["dims"] = dim

    ax.add_collection3d(cube.get_polycollection())
    space["ax"] = ax

    return space

# Get the axes object and abstract it as a 3D space.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_xlabel('z')

space = {"ax": ax, "dims": [None, None, None]}


# 'Cross-hair' block
base = np.array((0.5, 1, -0.5))
l_vec, w_vec, h_vec = np.array((0, 4, 0)), np.array((2, 0, 0)), np.array((0, 0, 2))
cross_hair_block = format_cube(construct_cube(base, l_vec, w_vec, h_vec), facecolours='r', linewidths=0.1, edgecolours='black', alpha=1)
space = add_cube_to_space(space, cross_hair_block)

# 'Peripheral' block (left)
base = np.array((-1, 3, -1))
l_vec, w_vec, h_vec = np.array((0, 1, 0)), np.array((1, 0, 0)), np.array((0, 0, 3))
peripheral_block_left = format_cube(construct_cube(base, l_vec, w_vec, h_vec), facecolours='b', linewidths=0, edgecolours='black', alpha=1) 
space = add_cube_to_space(space, peripheral_block_left)

# 'Peripheral' block (right)
base = np.array((3, 3, -1))
l_vec, w_vec, h_vec = np.array((0, 1, 0)), np.array((1, 0, 0)), np.array((0, 0, 3))
peripheral_block_right = format_cube(construct_cube(base, l_vec, w_vec, h_vec), facecolours='b', linewidths=0, edgecolours='black', alpha=1) 
space = add_cube_to_space(space, peripheral_block_right)

# 'Peripheral' block (lower)
base = np.array((-1, 3, -2))
l_vec, w_vec, h_vec = np.array((0, 1, 0)), np.array((5, 0, 0)), np.array((0, 0, 1))
peripheral_block_lower = format_cube(construct_cube(base, l_vec, w_vec, h_vec), facecolours='b', linewidths=0, edgecolours='black', alpha=1) 
space = add_cube_to_space(space, peripheral_block_lower)

# 'Peripheral' block (upper)
base = np.array((-1, 3, 2))
l_vec, w_vec, h_vec = np.array((0, 1, 0)), np.array((5, 0, 0)), np.array((0, 0, 1))
peripheral_block_upper = format_cube(construct_cube(base, l_vec, w_vec, h_vec), facecolours='b', linewidths=0, edgecolours='black', alpha=1) 
space = add_cube_to_space(space, peripheral_block_upper)

ax = space["ax"]
ax.set_xlim(space["dims"][0])
ax.set_ylim(space["dims"][1])
ax.set_zlim(space["dims"][2])

plt.show()
