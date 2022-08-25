import numpy


_shape3d = (132, 80, 114)  # TODO: Not hard coded..
_voxel_dimensions = (100, 100, 100)  # TODO: not hard coded


class CachedProjections(object):
    """
    This is a helper class to access the Allen Institute's mouse voxel connectivity data that fulfills two purposes:
    1. It returns the data in a slightly more usable format.
    Use CachedProjections.projection(list_of_source_regions, list_of_target_regions) to get a 2d connectivity array
    and the 3d coordinates of the source and target voxels
    2. It caches everything it looks up into an hdf5 file. This is useful because instantiating a "voxel_array" takes
    about 10 minutes, while opening an hdf5 file is instant, and the Allen Data might go offline at some point.
    Therefore, this class can run completely on a cache, without ever using the underlying Allen Institute data.
    """

    def __init__(self, allen_data=None, cache_file=None, grow_cache=True):
        """
        Initialize a CachedProjections object
        :param: allen_data: Either a VoxelModelCache object (see mcmodels) or a tuple comprising:
        (VoxelConnectivityArray, source_coords (numpy.array; N x 3), target_coords (numpy.array; M x 3),
        annotation_volume (numpy.array; N x M), StructureTree). When in doubt, just provide the VoxelModelCache.
        If neither the tuple not the VoxelModelCache are provided, the class will try to work entirely on the hdf5 cache

        :param: cache_file: Path that specifies where to put the hdf5 cache file. If the file does not exist, it will
        be created. Default: "projection_cache.h5"
        :param: grow_cache: If set to false, newly looked up projections are not written back into the cache. Useful
        if you worry running out of disk space
        """
        if allen_data is not None:
            if isinstance(allen_data, tuple):
                voxel_array, source_coords, target_coords, annotations, tree = allen_data
            else:
                voxel_array, source_coords, target_coords, annotations, tree = self.from_voxel_model_cache(allen_data)
            self.voxel_array = voxel_array
            self.source_3d = source_coords
            self.target_3d = target_coords
            self.vol = annotations
            self.tree = tree
            self._make_indices()
            self._cache_only = False
        else:
            self._cache_only = True
        if cache_file is None:
            self._cache_fn = "projection_cache.h5"
        else:
            self._cache_fn = cache_file
        self._grow_cache = grow_cache

    @staticmethod
    def from_voxel_model_cache(vmc):
        voxel_array, source_mask, target_mask = vmc.get_voxel_connectivity_array()
        source_3d = source_mask.coordinates
        target_3d = target_mask.coordinates  # TODO: Option to limit to right hemisphere..?
        vol, _ = vmc.get_annotation_volume()
        tree = vmc.get_structure_tree()
        return voxel_array, source_3d, target_3d, vol, tree

    def _make_indices(self):
        self.source_3d_flat = self._three_d_to_three_d_flat(self.source_3d)
        self.target_3d_flat = self._three_d_to_three_d_flat(self.target_3d)

    @staticmethod
    def _three_d_to_three_d_flat(idx):
        return idx[:, 0] * numpy.prod(_shape3d[1:]) + \
               idx[:, 1] * _shape3d[-1] + idx[:, 2]

    def _three_d_indices_to_output_coords(self, idx, direction):
        return numpy.array(idx) * numpy.array([_voxel_dimensions])

    def _three_d_flat_to_array_flat(self, three_d_flat, index_as, strict=False):
        if strict:
            assert numpy.all(numpy.diff(three_d_flat) > 0)
        if index_as == 'source':
            if strict:
                assert numpy.all(numpy.in1d(three_d_flat, self.source_3d_flat))
            return numpy.nonzero(numpy.in1d(self.source_3d_flat, three_d_flat))[0]
        elif index_as == 'target':
            if strict:
                assert numpy.all(numpy.in1d(three_d_flat, self.target_3d_flat))
            return numpy.nonzero(numpy.in1d(self.target_3d_flat, three_d_flat))[0]
        else:
            raise Exception("Need to index as either 'source', or 'target'")

    def _region_ids(self, regions, resolve_to_leaf=False):
        if not isinstance(regions, list) or isinstance(regions, numpy.ndarray):
            regions = [regions]
        r_struc = self.tree.get_structures_by_acronym(regions)
        r_ids = numpy.array([_x['id'] for _x in r_struc])

        def resolver(r_ids):
            rslvd = [resolver(_chldr) if len(_chldr) else _base
                     for _base, _chldr in
                     zip(r_ids, self.tree.child_ids(r_ids))]
            return numpy.hstack(rslvd)

        if resolve_to_leaf:
            return resolver(r_ids)
        return r_ids

    def _make_volume_mask(self, idxx):
        return numpy.in1d(self.vol.flat, idxx).reshape(self.vol.shape)

    def make_volume_mask(self, regions):
        idxx = self._region_ids(regions, resolve_to_leaf=True)
        return self._make_volume_mask(idxx)

    def mask_to_indices(self, vol_mask, index_as):
        mask_idx = numpy.nonzero(vol_mask)
        mask_idx = numpy.vstack(mask_idx).transpose()
        mask_idx_flat = self._three_d_to_three_d_flat(mask_idx)
        if index_as == 'source':
            some_3d_flat = self.source_3d_flat
        elif index_as == 'target':
            some_3d_flat = self.target_3d_flat
        else:
            raise Exception("Need to index as either 'source', or 'target'")
        valid = numpy.in1d(mask_idx_flat, some_3d_flat)
        mask_idx = mask_idx[valid]
        mask_idx_flat = mask_idx_flat[valid]
        return mask_idx, \
               self._three_d_flat_to_array_flat(mask_idx_flat, index_as)

    def indices_for_region(self, regions, index_as):
        mask = self.make_volume_mask(regions)
        return self.mask_to_indices(mask, index_as)

    def _uncached_projection(self, src_regions, tgt_regions):
        if self._cache_only:
            raise ValueError("Working in cache-only mode. Please use .projection!")
        src_3d, src_array = self.indices_for_region(src_regions, 'source')
        tgt_3d, tgt_array = self.indices_for_region(tgt_regions, 'target')
        values = self.voxel_array[src_array, tgt_array]
        return values, src_3d, tgt_3d

    def uncached_projection(self, src_regions, tgt_regions):
        values, src_coord, tgt_coord = self._uncached_projection(self, src_regions, tgt_regions)
        src_coord = self._three_d_indices_to_output_coords(src_coord, 'source')
        tgt_coord = self._three_d_indices_to_output_coords(tgt_coord, 'target')
        return values, src_coord, tgt_coord

    def _cached_single_projection(self, src_region, tgt_region, coordinate_name='3d', write_cache=True):
        import h5py
        expected_grp = "{0}/{1}".format(src_region, tgt_region)
        with h5py.File(self._cache_fn, "a") as h5:
            if expected_grp in h5:
                grp = h5[expected_grp]
                return (grp['data'][:],
                        grp[coordinate_name]['source_coordinates'][:],
                        grp[coordinate_name]['target_coordinates'][:])
            elif self._cache_only:
                raise ValueError("Working in cache-only mode, but {0} to {1} is not in cache!".format(src_region,
                                                                                                      tgt_region))
            else:
                V, src, tgt = self._uncached_projection([src_region], [tgt_region])
                if write_cache:
                    grp = h5.require_group(expected_grp)
                    grp.create_dataset('data', data=V)
                    grp = grp.create_group(coordinate_name)
                    grp.create_dataset('source_coordinates', data=src)
                    grp.create_dataset('target_coordinates', data=tgt)
                return V, src, tgt

    def projection(self, src_regions, tgt_regions, coordinate_name='3d'):
        """
        Returns the voxel coordinates and projection strengths for a specified projection
        :param: src_regions: list of acronyms of source regions of the projection
        :param: tgt_regions: list of acronyms of target regions of the projection
        :param: coordinate_name: For future functionality. Do not use!
        """
        assert len(src_regions) > 0 and len(tgt_regions) > 0
        all_src_coord = []
        all_V = []
        for src in src_regions:
            V, src_coord, tgt_coord = zip(*[self._cached_single_projection(src, tgt, coordinate_name=coordinate_name,
                                                                           write_cache=self._grow_cache)
                                            for tgt in tgt_regions])
            V = numpy.hstack(V)
            all_src_coord.append(src_coord[0])  # TODO: Check consistency of src_coord
            all_V.append(V)
        #  TODO: Check consistency of tgt_coord
        src_coord = self._three_d_indices_to_output_coords(numpy.vstack(all_src_coord), 'source')
        tgt_coord = self._three_d_indices_to_output_coords(numpy.vstack(tgt_coord), 'target')
        return numpy.vstack(all_V), src_coord, tgt_coord

