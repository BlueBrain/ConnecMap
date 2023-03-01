import voxcell


class GeneralMap(object):

    def __init__(self, *args, **kwargs):
        self._maps = args
        if len(args) > 0:
            for i in range(len(args) - 1):
                assert args[i].dim_out == args[i + 1].dim_in or args[i + 1].dim_in == -1
            self.dim_in = args[0].dim_in
            self.dim_out = args[-1].dim_out
        else:
            self.dim_in = kwargs.get("dim_in", -1)
            self.dim_out = kwargs.get("dim_out", -1)

    def __payload__(self, coords):
        for mp in self._maps:
            coords = mp(coords)
        return coords

    def __call__(self, coords):
        assert coords.shape[1] == self.dim_in or self.dim_in == -1
        return self.__payload__(coords)

    def __add__(self, other):
        return GeneralMap(self, other)


class BarycentricMap(GeneralMap):
    def __init__(self, bary_obj):
        self._bary = bary_obj
        super(BarycentricMap, self).__init__(dim_in=self._bary._dim, dim_out=3)

    def __payload__(self, coords):
        return self._bary.col(*coords.transpose())


class Flatmap(GeneralMap):

    def __init__(self, fn_nrrd):
        self._volume = voxcell.VoxelData.load_nrrd(fn_nrrd)
        super(Flatmap, self).__init__(dim_in=len(self._volume.shape), dim_out=self._volume.payload_shape[0])

    def __payload__(self, coords):
        return self._volume.lookup(coords)




