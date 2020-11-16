import numpy as np
import itertools


def adjacency_mask(image=None, shape=None, mask_type="orthogonal"):
    if image:
        img_shape = np.shape(image)[
            :2
        ]  # Only 2d images and we assume ev. channel is last index
    if shape:
        inp_shape = shape
    if (image and shape) and inp_shape == img_shape:
        shape = inp_shape
    elif (image and shape) and inp_shape != img_shape:
        raise ValueError("The provided image does not agree with the stated shape.")
    elif image and not shape:
        shape = img_shape
    elif shape and not image:
        shape = inp_shape
    elif not image and not shape:
        raise ValueError("You need to enter either an image or an image shape.")

    rows = shape[0]
    cols = shape[1]
    mask = np.zeros((rows * cols, rows * cols))
    for ii in range(rows * cols):
        index = index_flat_to_array(ii, shape)
        neighbours = get_neighbours(index, shape, neighbour_type=mask_type)
        mask[ii, :].flat[neighbours] = 1

    return mask


def get_neighbours(
    index, shape, neighbour_type="orthogonal", out_type="flat", flatten_type="C"
):
    row, col = index[0], index[1]
    rows, cols = shape[0], shape[1]

    if row == 0:
        row_neighbours = [row, row + 1]
    elif row == rows - 1:
        row_neighbours = [row - 1, row]
    else:
        row_neighbours = [row - 1, row, row + 1]

    if col == 0:
        col_neighbours = [col, col + 1]
    elif col == cols - 1:
        col_neighbours = [col - 1, col]
    else:
        col_neighbours = [col - 1, col, col + 1]

    neighbours = list(itertools.product(row_neighbours, col_neighbours))

    if neighbour_type == "orthogonal":
        rem = []
        for neighbour in neighbours:
            if manhattan_distance(index, neighbour) > 1:
                rem.append(neighbour)
        for coord in rem:
            neighbours.remove(coord)
    if out_type == "flat":
        neighbours = index_array_to_flat(neighbours, shape, flatten_type=flatten_type)
    return neighbours


def manhattan_distance(p1, p2):
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])


def index_flat_to_array(flat_index_inp, shape, flatten_type="C"):
    if not isinstance(flat_index_inp, list):
        not_list = True
        flat_index_inp = [flat_index_inp]
    else:
        not_list = False
    index = []

    for flat_index in flat_index_inp:
        rows = shape[0]
        cols = shape[1]
        if flat_index >= rows * cols:
            raise ValueError(
                "The provided flat index is out of bounds for the provided shape."
            )
        if flatten_type == "C":
            r, c = flat_index // cols, flat_index % cols
        elif flatten_type == "R":
            r, c = flat_index % cols, flat_index // cols
        else:
            raise ValueError
        index.append((r, c))
    if not_list:
        index = index[0]
    return index


def index_array_to_flat(array_index_inp, shape, flatten_type="C"):
    if not isinstance(array_index_inp, list):
        not_list = True
        array_index_inp = [array_index_inp]
    else:
        not_list = False
    index = []
    for array_index in array_index_inp:
        r, c = array_index
        rows = shape[0]
        cols = shape[1]
        if r >= rows or c >= cols:
            raise ValueError(
                "The set array index is out of bounds for the provided shape."
            )
        if flatten_type == "C":
            flat_index = r * cols + c
        elif flatten_type == "R":
            flat_index = c * rows + r
        else:
            raise ValueError
        index.append(flat_index)

    if not_list:
        index = index[0]
    return index
