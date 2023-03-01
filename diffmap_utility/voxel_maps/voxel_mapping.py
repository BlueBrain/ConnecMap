import numpy


def innervation_location(vol, src_coordinates, tgt_coordinates, view='target', method='mean'):
    """
    Calculates for each point in a volume the average location it is innervated from
    :param: vol (numpy.array; shape: N x M) Array of innervation strength values shape: (source voxels x target voxels)
    :param: src_coordinates (numpy.array; shape N x [2/3]) The coordinates of projection source voxels in any two-
    or three-dimensional coordinate system.
    :param: tgt_coordinates (numpy.array; shape m x [2/3]) Same, for coordinates of target voxels.
    :param: view (default: 'target') If view is "target" the innervation will be averaged along source voxels.
    If view is 'source' it will be averaged along target voxels.
    :param: method (str): Method of averaging. Either 'mean' or 'normalize'
    """
    if view == 'source':
        return innervation_location(vol.transpose(), tgt_coordinates, src_coordinates, view='target', method=method)
    elif view != 'target':
        raise ValueError()
    innervation = numpy.dot(vol.transpose(), src_coordinates)
    if method == 'mean':
        innervation = innervation / vol.sum(axis=0, keepdims=True).transpose()
    elif method == 'normalize':
        innervation = innervation / innervation.sum(axis=1, keepdims=True)
    elif method == 'manual':
        return innervation, vol.sum(axis=0), tgt_coordinates
    else:
        raise ValueError()
    return innervation, tgt_coordinates


def coordinates_to_image(data, coordinates, counts=None, method='mean', shape=None):
    if counts is None:
        counts = numpy.ones(data.shape[0])
    coords = numpy.round(coordinates).astype(int)
    if shape is None:
        shape = coords.max(axis=0) + 1

    img_out = numpy.zeros(tuple(shape) + (data.shape[1], ), dtype=float)
    counts_out = numpy.zeros(tuple(shape) + (1, ), dtype=float)

    for out_data, out_count, out_coord in zip(data, counts, coords):
        img_indices = numpy.ix_(*list(map(lambda x: [x], out_coord)))
        img_out[img_indices] += out_data
        counts_out[img_indices] += out_count

    if method == 'mean':
        img_out = img_out / counts_out
    elif method == 'normalize':
        img_out = img_out / img_out.sum(axis=2, keepdims=True)
    elif method == 'manual':
        return img_out, counts_out
    else:
        raise ValueError()
    return img_out


def innervation_image(vol, src_coordinates, tgt_coordinates, view='target', method='mean', shape=None):
    """
    Calculates for each point in a volume the average location it is innervated from and puts the results into
    an array that is indexed by the provided coordinates of the volume. If those coordinates are 2d, you can
    visualize the result as an image.
    :param: vol (numpy.array; shape: N x M) Array of innervation strength values shape: (source voxels x target voxels)
    :param: src_coordinates (numpy.array; shape N x [2/3]) The coordinates of projection source voxels in any two-
    or three-dimensional coordinate system.
    :param: tgt_coordinates (numpy.array; shape m x [2/3]) Same, for coordinates of target voxels.
    :param: view (default: 'target') If view is "target" the innervation will be averaged along source voxels, and
    the resulting array will be indexed by target coordinates. If view is 'source' it will be averaged along target
    voxels and the resulting array will be indexed by source coordinates.

    Tip: If the voxels you are averaging along are associated with 3d coordinates and the other set of voxels is
    associated with 2d coordinates, then the result can be visualized as an image!

    :param: method (str): Method of averaging. Either 'mean' or 'normalize'
    :param: shape (tuple): The shape of the output array. Must be larger than the largest coordinate of the voxels
    used and have the same dimensionality
    """
    innervation, counts, coords = innervation_location(vol, src_coordinates, tgt_coordinates, view=view,
                                                       method='manual')
    return coordinates_to_image(innervation, coords, counts=counts, method=method, shape=shape)

