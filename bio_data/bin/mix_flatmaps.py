import voxcell
import numpy


def load_flatmaps(mixer):
    fm_dict = {}
    shape = None
    voxel_dims = None
    offset = None
    for k in mixer.keys():
        if k not in fm_dict:
            fm = voxcell.VoxelData.load_nrrd(mixer[k])
            if shape is not None:
                assert shape == fm.raw.shape
                assert numpy.all(voxel_dims == fm.voxel_dimensions)
                assert numpy.all(offset == fm.offset)
            shape = fm.raw.shape
            voxel_dims = fm.voxel_dimensions
            offset = fm.offset
            fm_dict[k] = voxcell.VoxelData.load_nrrd(mixer[k])
        mixer[k] = fm_dict[k]
    return shape, voxel_dims, offset


def get_mask_for_regions(ann, hier, lst_regions):
    mask = (~numpy.isnan(ann.raw)) & (ann.raw > -1)

    if lst_regions is not None:
        reg_ids = [list(hier.collect("acronym", reg_name, "id")) for reg_name in lst_regions]
        mask = mask & (numpy.in1d(ann.raw.flat, numpy.hstack(reg_ids)).reshape(ann.raw.shape))
    return mask


def mix(ann, hier, mixer, tgt_shape, voxel_dims, offset):
    raw = -numpy.ones(tgt_shape, dtype=int)
    for k, fm in mixer.items():
        mask = get_mask_for_regions(ann, hier, [k])
        raw[mask] = fm.raw[mask]
    return voxcell.VoxelData(raw, voxel_dimensions=voxel_dims, offset=offset)


if __name__ == "__main__":
    import argparse, os, json
    parser = argparse.ArgumentParser(description='Mix a flatmap using entries from different flatmaps\n')
    parser.add_argument('--input_path', help='Input mixing recipe')
    parser.add_argument('--output_path', help='Flatmap output path', default='./mixed_flatmap.nrrd')
    parser.add_argument('--hierarchy', help='Json file that specifies the region hierarchy')
    parser.add_argument('--annotation', help='Nrrd file that specifies the brain region ids')
    args = parser.parse_args()

    output_path = args.output_path

    assert os.path.isfile(args.input_path), "Cannot find input flatmap: {0}".format(str(args.input_path))
    assert os.path.isfile(args.hierarchy), "Cannot find hierarchy file: {0}".format(str(args.hierarchy))
    assert os.path.isfile(args.annotation), "Cannot find annotation file: {0}".format(str(args.annotation))

    with open(args.input_path, "r") as fid:
        mixer = json.load(fid)
    shape, voxel_dims, offset = load_flatmaps(mixer)
    hier = voxcell.Hierarchy.load_json(args.hierarchy)
    ann = voxcell.VoxelData.load_nrrd(args.annotation)

    mix(ann, hier, mixer, shape, voxel_dims, offset).save_nrrd(output_path)
