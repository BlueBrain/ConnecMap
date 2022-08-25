import numpy
import logging
from matplotlib import pyplot as plt

info_log = logging.getLogger(__file__)


class BarycentricCoordinates(object):

    def __init__(self, *args):
        """
        :param args: x, y and z coordinates of three points, i.e. cls(x, y, z)
        or x and y coordinates in the two-dimensional case, i.e. cls(x, y)
        """
        assert numpy.all(numpy.array(list(map(len, args))) == 3)
        self._dim = len(args)
        assert self._dim >= 2 and self._dim <= 3
        self._x = args[0]
        self._y = args[1]
        self._z = None
        if self._dim > 2:
            self._z = args[2]
        self._initialize_matrices()

    def _initialize_matrices(self):
        if self._dim == 2:
            self._coords = (self._x, self._y)
        else:
            self._coords = (self._x, self._y, self._z)
        self._S = numpy.matrix(self._coords).transpose()
        self._T = numpy.matrix(self._transform_matrix())

    def _transform_matrix(self):
        T = numpy.array([[_c[0] - _c[2], _c[1] - _c[2]]
                         for _c in self._coords])
        # In 3-d case we append the normal of the defined plane. In conversions we will ignore this third coordinate
        if self._dim == 3:
            N = numpy.cross(T[:, 0], T[:, 1])
            N = N / numpy.linalg.norm(N)
            T = numpy.hstack([T, numpy.vstack(N)])
        return T
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def z(self):
        return self._z
    
    @x.setter
    def x(self, newval):
        self._x = newval
        self._initialize_matrices()
    
    @y.setter
    def y(self, newval):
        self._y = newval
        self._initialize_matrices()
    
    @z.setter
    def z(self, newval):
        assert self._dim > 2
        self._z = newval
        self._initialize_matrices()

    def cart2bary(self, *args):
        assert len(args) == self._dim
        lc = [c - b[2] for c, b in zip(args, self._coords)]
        res = numpy.linalg.solve(self._T, numpy.vstack(lc))
        # If it's the 3-d case this is where we ignore the normal component
        return numpy.vstack([res[0, :], res[1, :], 1.0 - res[:2, :].sum(axis=0)]).transpose()

    def bary2cart(self, a, b, c):
        return numpy.array(numpy.vstack([a, b, c]).transpose() * self._S)

    def area(self):
        p1 = numpy.array([_c[0] - _c[2] for _c in self._coords])
        p2 = numpy.array([_c[1] - _c[2] for _c in self._coords])
        l1 = numpy.linalg.norm(p1)
        l2 = numpy.linalg.norm(p2)
        assert l1 > 0 and l2 > 0
        return numpy.sin(numpy.arccos(numpy.sum(p1 * p2) / (l1 * l2))) * (l1 * l2) * 0.5


class BarycentricFlatmap(BarycentricCoordinates):
    """A barycentric coordinate system that in the 3d case additionally provides the implied flatmap.
    That is, the flatmap defined as follows: For a 3d point, first parallel-project it into the plane
    of the barycentric triangle. Then convert it to the orthonormal base given by its first base vector and
    an orthogonal vector in the plane."""
    def __init__(self, *args):
        super(BarycentricFlatmap, self).__init__(*args)
        if self._dim == 3:
            self.implied_flatmap = self.__implied_flatmap_3d
        else:
            self.implied_flatmap = self.__implied_flatmap_2d

    def _initialize_flatmap(self):
        pO = [_c[2] for _c in self._coords]
        T = numpy.array(self._T)
        v1 = T[:, 0]
        v2 = T[:, 1]
        v1 = v1 / numpy.linalg.norm(v1)
        N = numpy.cross(v1, v2)
        N = N / numpy.linalg.norm(N)
        v2 = numpy.cross(v1, N)
        self._flatmapper = BarycentricCoordinates([pO[0] + v1[0], pO[0] + v2[0], pO[0]],
                                                  [pO[1] + v1[1], pO[1] + v2[1], pO[1]],
                                                  [pO[2] + v1[2], pO[2] + v2[2], pO[2]])
    
    def _initialize_matrices(self):
        super(BarycentricFlatmap, self)._initialize_matrices()
        if self._dim == 3:
            self._initialize_flatmap()

    def __implied_flatmap_3d(self, *args):
        out = self._flatmapper.cart2bary(*args)
        return out[:, :2]

    def __implied_flatmap_2d(self, *args):
        return numpy.vstack(args).transpose()


class BarycentricColors(BarycentricFlatmap):

    # noinspection PyDefaultArgument
    def __init__(self, *args, **kwargs):
        super(BarycentricColors, self).__init__(*args)
        self._cols = numpy.matrix(numpy.vstack([kwargs.get('red', [1, 0, 0]),
                                                kwargs.get('green', [0, 1, 0]),
                                                kwargs.get('blue', [0, 0, 1])]).transpose())

    def col(self, *args):
        b = self.cart2bary(*args)
        b[b > 1.0] = 1.0
        b[b < 0.0] = 0.0
        return numpy.array((self._cols * b.transpose()).transpose())


class BarycentricConstrainedColors(BarycentricColors):

    def col(self, *args):
        from scipy.stats import norm
        b = self.cart2bary(*args)
        w = norm(-0.25, 0.175).cdf(numpy.min(b, axis=1)).reshape((len(b), 1))
        b = w * b
        b[b > 1.0] = 1.0
        b[b < 0.0] = 0.0
        return numpy.array((self._cols * b.transpose()).transpose())

    
class BarycentricMaskMapper(BarycentricColors):

    def __init__(self, mask, contract=0.75, **kwargs):
        self.__tmp_x = []
        self.__tmp_y = []
        self.__kwargs = kwargs
        self._mask = mask
        if mask is not None:
            self._nz = numpy.vstack(numpy.nonzero(self._mask)).transpose()
        self._find_corners(contract=contract)

    def _find_corners(self, contract=0.75):
        from scipy.spatial import distance, distance_matrix
        nz = self._nz
        cog = numpy.mean(nz, axis=0)

        D = distance_matrix([cog], nz)[0]
        p1 = numpy.argmax(D) # first point: Furthest away from center of region
        pD = distance.squareform(distance.pdist(nz)) # pairwise distance for all pairs
        p1D = distance_matrix(nz, nz[p1:(p1+1)]) # distance to first point. Shape: N x 1

        mx = numpy.argmax(pD + p1D + p1D.transpose()) # maximize sum of distances between the three points
        p2 = numpy.mod(mx, pD.shape[1])
        p3 = mx / pD.shape[1]
        ret = nz[[p1.astype(int), p2.astype(int), p3.astype(int)]]
        ret = cog + contract * (ret - cog)
        super(BarycentricMaskMapper, self).__init__(*ret.transpose(),  # ret[:, 0], ret[:, 1],
                                                    **self.__kwargs)

class IrregularGridMapper(BarycentricMaskMapper):

    def __init__(self, xy, contract=0.75, **kwargs):
        self._nz = xy
        super(IrregularGridMapper, self).__init__(None, contract=contract)

