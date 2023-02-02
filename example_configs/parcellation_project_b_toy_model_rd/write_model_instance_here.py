import numpy
import os
import sys
import voxcell
from projection_voxels import BinarySplitVoxelArray, BinarySplitModel
from projection_voxels.binary_split_model import RootRegion


if __name__ == "__main__":
    this_root = os.path.split(__file__)[0]
    sub_path = "input/configuration"
    if not os.path.isdir(os.path.join(this_root, sub_path)):
        os.makedirs(os.path.join(this_root, sub_path))

    seed = int(sys.argv[1])
    
    shape = [90, 100, 1]

    root = RootRegion(numpy.array([[0, shape[0]], [0, shape[1]]]))
    con_params = {
            "str_max": 4,
            "str_exponent": -1,
            "range_max": 25,
            "range_div": 3,
            "offset": 1,
            "noise": 0.2
        }

    bsp = BinarySplitVoxelArray(root, shape[2], con_params)
    bsp.initialize_split(seed)
    bsp.save_json(os.path.join(this_root, sub_path, BinarySplitModel.FN_MODEL))
    bsp.save_atlas(os.path.join(this_root, sub_path, BinarySplitModel.FN_ANNOTATIONS))
    bsp.save_hierarchy(os.path.join(this_root, sub_path, BinarySplitModel.FN_HIERARCHY))

    flatmap = numpy.zeros(tuple(shape) + (2,), dtype=int)
    flatmap[bsp.source_coords_3d[:, 0],
            bsp.source_coords_3d[:, 1],
            bsp.source_coords_3d[:, 2], :] = bsp.source_coords_3d[:, :2]
    vc_fm = voxcell.VoxelData(flatmap, (100, 100, 100), offset=(0.0, 0.0, 0.0))
    vc_fm.save_nrrd(os.path.join(this_root, sub_path, "anatomical_flatmap.nrrd"))
