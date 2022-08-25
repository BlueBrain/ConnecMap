import numpy
import voxcell


def mirror_fm(fm, ann, hier, lst_regions, index_to_offset_hemis=1, offset_to_use=None):
    from scipy.interpolate import LinearNDInterpolator
    pivot = int(ann.raw.shape[2] / 2)
    mask_right = get_mask_for_regions(ann, hier, lst_regions)
    mask_right[:, :, :pivot] = False
    mask_left = get_mask_for_regions(ann, hier, lst_regions)
    mask_left[:, :, pivot:] = False

    if offset_to_use is None:
        offset_to_use = numpy.nanmax(fm.raw[mask_right, index_to_offset_hemis]) + 1

    xyz_right = numpy.nonzero(mask_right)
    vals_right = fm.raw[mask_right]
    xyz_left = numpy.nonzero(mask_left)
    xyz_left = [xyz_left[0], xyz_left[1], 2 * pivot - 1 - xyz_left[2]]

    print("\t...Mirroring at pivot {0}".format(offset_to_use))
    for i in range(vals_right.shape[1]):
        ip = LinearNDInterpolator(list(zip(*xyz_right)), vals_right[:, i], fill_value=-1)
        ip_vals = ip(*xyz_left)
        fm.raw[mask_left, i] = ip_vals
    # Move right hemisphere to right half of flat map
    fm.raw[mask_right, index_to_offset_hemis] = 2 * offset_to_use - fm.raw[mask_right, index_to_offset_hemis] - 1


def normalize_percentile(data):
    _percs = numpy.arange(0, 101)
    tmp = numpy.vstack([numpy.interp(_d, numpy.percentile(_d, _percs), _percs)
                        for _d in data.transpose()])
    tmp = data.max(axis=0, keepdims=True) * tmp.transpose() / 100
    return tmp.astype(int)


def get_mask_for_regions(ann, hier, lst_regions):
    mask = (~numpy.isnan(ann.raw)) & (ann.raw > -1)

    if lst_regions is not None:
        reg_ids = [list(hier.collect("acronym", reg_name, "id")) for reg_name in lst_regions]
        mask = mask & (numpy.in1d(ann.raw.flat, numpy.hstack(reg_ids)).reshape(ann.raw.shape))
    return mask


def normalize_coordinates_for_regions(fm, ann, hier, lst_regions):
    mask = get_mask_for_regions(ann, hier, lst_regions)
    mask = mask & numpy.all((~numpy.isnan(fm.raw)) & (fm.raw > -1), axis=3)
    fm.raw[mask] = normalize_percentile(fm.raw[mask])


def convert_to_int(fm):
    fm.raw[numpy.isnan(fm.raw)] = -1
    fm.raw = fm.raw.astype(int)


def treat_fm(fm, ann, hier, lst_lst_regions, **kwargs):
    for lst_reg in lst_lst_regions:
        print("Treating regions: {0}".format(str(lst_reg)))
        try:
            pass
            #  normalize_coordinates_for_regions(fm, ann, hier, lst_reg)
        except:
            print("... Error during normalization!")
        try:
            mirror_fm(fm, ann, hier, lst_reg, **kwargs)
        except:
            print("... Error during mirroring!")
        try:
            convert_to_int(fm)
        except:
            print("... Error during conversion to int data type!")


if __name__ == "__main__":
    import argparse, os, json

    index_to_offset_hemis = 1

    parser = argparse.ArgumentParser(description='Post-process a diffusion based flatmap\n')
    parser.add_argument('--input_path', help='Flatmap input path')
    parser.add_argument('--output_path', help='Flatmap output path', default='./pp_flatmap.nrrd')
    parser.add_argument('--region_file', help='Json file that specifies which regions to flatten')
    parser.add_argument('--hierarchy', help='Json file that specifies the region hierarchy')
    parser.add_argument('--annotation', help='Nrrd file that specifies the brain region ids')
    args = parser.parse_args()

    output_path = args.output_path

    assert os.path.isfile(args.input_path), "Cannot find input flatmap: {0}".format(str(args.input_path))
    assert os.path.isfile(args.region_file), "Cannot find region file: {0}".format(str(args.region_file))
    assert os.path.isfile(args.hierarchy), "Cannot find hierarchy file: {0}".format(str(args.hierarchy))
    assert os.path.isfile(args.annotation), "Cannot find annotation file: {0}".format(str(args.annotation))

    with open(args.region_file, "r") as fid:
        regions = json.load(fid)
    hier = voxcell.Hierarchy.load_json(args.hierarchy)
    ann = voxcell.VoxelData.load_nrrd(args.annotation)
    fm = voxcell.VoxelData.load_nrrd(args.input_path)

    offset_to_use = numpy.nanmax(fm.raw[:, :, :, index_to_offset_hemis]) + 1

    if not isinstance(regions, list):
        regions = [regions]
    lst_lst_regions = [_regions["flatten"] for _regions in regions]

    treat_fm(fm, ann, hier, lst_lst_regions, index_to_offset_hemis=index_to_offset_hemis, offset_to_use=offset_to_use)
    fm.save_nrrd(args.output_path)
