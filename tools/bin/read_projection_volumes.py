import h5py, numpy
from barycentric import BarycentricMaskMapper
from skimage.color import rgb2hsv


_shape3d = (132, 80, 114)
_shape2d = (136, 272)
_co_hemi3d = _shape3d[2] / 2
_co_hemi2d = _shape2d[1] / 2
_str_src = "source_coordinates"
_str_tgt = "target_coordinates"


def read_volume(h5, lst_src, lst_tgt):
    V = numpy.vstack([numpy.hstack([h5[_src][_tgt]["data"]
                                    for _tgt in lst_tgt])
                     for _src in lst_src])
    return V


def read_coordinates(h5, lst_src, lst_tgt, which_format, which_end):
    if which_end == _str_tgt:
        return numpy.vstack([h5[lst_src[0]][_tgt][which_format][_str_tgt] for _tgt in lst_tgt])
    elif which_end == _str_src:
        return numpy.vstack([h5[_src][lst_tgt[0]][which_format][_str_src] for _src in lst_src])
    else:
        raise Exception()


def idx2mask(idx):
    idx = idx[~numpy.any(numpy.isnan(idx), axis=1)]
    idx = numpy.round(idx).astype(int)
    out = numpy.zeros(_shape2d, dtype=bool)
    out[idx[:, 0], idx[:, 1]] = True
    return out


def _mask_hemisphere3d(coord3d, V):
    valid = coord3d[:, 2] >= _co_hemi3d
    V = V[:, valid]
    coord3d = coord3d[valid]
    return coord3d, V


def _mask_hemisphere2d(coord2d, V):
    valid = coord2d[:, 1] >= _co_hemi2d
    V = V[:, valid]
    coord2d = coord2d[valid]
    return coord2d, V


def _project_and_normalize_colors(V, cols, saturation_co=1.0, rel_co_perc=95, rel_co_fac=0.25):
    saturation = rgb2hsv([cols])[0][:, 1]
    valid = saturation >= saturation_co
    V = V[:, valid]; cols = cols[valid, :]
    projected = numpy.dot(V, cols)
    nrmlz = V.sum(axis=1, keepdims=True)
    co = numpy.percentile(nrmlz, rel_co_perc) * rel_co_fac
    nrmlz = numpy.maximum(nrmlz, co)
    return projected / nrmlz


# Representation CT: 4d volume, 132 x 80 x 114 x 3: X x Y x Z x C
def volume3d(h5, lst_src, lst_tgt, direction='forward', show_img=False):
    V = read_volume(h5, lst_src, lst_tgt)
    if direction == 'forward':
        coord2d = read_coordinates(h5, lst_src, lst_tgt, "2d", _str_src)
        coord3d = read_coordinates(h5, lst_src, lst_tgt, "3d", _str_tgt)
        coord3d, V = _mask_hemisphere3d(coord3d, V)
        V = V.transpose()
    elif direction == 'backward':
        coord2d = read_coordinates(h5, lst_src, lst_tgt, "2d", _str_tgt)
        coord3d = read_coordinates(h5, lst_src, lst_tgt, "3d", _str_src)
        coord2d, V = _mask_hemisphere2d(coord2d, V)
    else:
        raise Exception()

    bary = BarycentricMaskMapper(idx2mask(coord2d), interactive=False, contract=0.95, show_img=show_img)
    cols = bary.col(coord2d[:, 0], coord2d[:, 1])
    cols[numpy.isnan(cols)] = 0
    vol = numpy.zeros(_shape3d + (3,), dtype=float)
    vol[coord3d[:, 0], coord3d[:, 1], coord3d[:, 2], :] = _project_and_normalize_colors(V, cols)

    return vol


# Representation TC: 3d volume: N x 136 x 272 & N x 3: N x fX x fY & N x (x, y, z)
def volume2d(h5, lst_src, lst_tgt, direction='forward'):
    V = read_volume(h5, lst_src, lst_tgt)
    coord2d = read_coordinates(h5, lst_src, lst_tgt, "2d", _str_tgt)
    coord2d, V = _mask_hemisphere2d(coord2d, V)
    coord2d = numpy.round(coord2d).astype(int)
    coord3d = read_coordinates(h5, lst_src, lst_tgt, "3d", _str_src)
    vol_out = numpy.NaN * numpy.ones((V.shape[0],) + _shape2d)
    for i, row in enumerate(V):
        vol_out[i][coord2d[:, 0], coord2d[:, 1]] = row
    return vol_out, coord3d


def plot_projection_3d(h5, lst_src, lst_tgt, direction='forward'):
    from mpl_toolkits.mplot3d import Axes3D
    V = volume3d(h5, lst_src, lst_tgt, direction=direction)
    nz = numpy.nonzero(numpy.nansum(V, axis=-1))
    ax = plt.figure().gca(projection='3d')
    for x, y, z in zip(*nz):
        col = V[x, y, z]
        ax.plot([x], [y], [z], 's', ms=15, color=col, alpha=1.5*numpy.mean(col))


