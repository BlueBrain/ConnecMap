#!/usr/bin/env python
import h5py
import numpy

from diffusion_mapping import flatten_pathway
from projection_voxels import projection_from_json

default_vxl_shape = (132, 80, 114)
default_vxl_offset = numpy.array([0., 0., 0.])
default_vxl_dims = numpy.array([100., 100., 100.])


def write_flatmap(flattened_regions, considered_regions, three_d_coords, flatmap_coords, output_path,
                  vxl_dims=None, vxl_shape=None):
    flat_str = ",".join(sorted(flattened_regions))
    cons_str = ",".join(sorted(considered_regions))
    from voxcell import VoxelData

    if os.path.isfile(output_path):
        existing_map = VoxelData.load_nrrd(output_path)
        vxl_dims = existing_map.voxel_dimensions
        vxl_offset = existing_map.offset
        vxl_shape = existing_map.shape
        existing_map = existing_map.raw
    else:
        if vxl_dims is None:
            vxl_dims = default_vxl_dims
        if vxl_shape is None:
            vxl_shape = default_vxl_shape
        vxl_offset = default_vxl_offset
        existing_map = numpy.NaN * numpy.ones(vxl_shape + (flatmap_coords.shape[1],), dtype=float)
    
    three_d_coords = (three_d_coords / vxl_dims).astype(int)  # TODO: Better way
    existing_map[three_d_coords[:, 0], three_d_coords[:, 1], three_d_coords[:, 2]] = flatmap_coords

    print("Writing flatmap for regions {src} via {tgt} to {fn}".format(src=flat_str,
                                                                       tgt=cons_str,
                                                                       fn=output_path)
          )

    my_map = VoxelData(existing_map, voxel_dimensions=tuple(vxl_dims),
                       offset=tuple(vxl_offset))  # map in .nrrd format
    VoxelData.save_nrrd(my_map, output_path)


def relax_positions(coords, reference_coords, dist_cutoff=101., n_iter=100):
    from sknetwork.embedding import Spring
    from scipy import sparse
    from scipy.spatial import distance

    D = distance.squareform(distance.pdist(reference_coords))
    G = sparse.coo_matrix((D > 0) & (D < dist_cutoff))
    spr = Spring(n_components=3, n_iter=n_iter, tol=1E-6)
    res = spr.fit_transform(G, position_init=coords)
    return res


def align_positions(coords, reference, normalize=False, resolution=100):
    from scipy.spatial import transform
    
    c_nrml = (coords - coords.mean(axis=0, keepdims=True)) / coords.std(axis=0, keepdims=True)
    r_nrml = (reference - reference.mean(axis=0, keepdims=True)) / reference.std(axis=0, keepdims=True)
    
    if normalize:
        c_nrml = c_nrml / numpy.linalg.norm(c_nrml, axis=1, keepdims=True)
        r_nrml = r_nrml / numpy.linalg.norm(r_nrml, axis=1, keepdims=True)
    
    rot_tf, _ = transform.Rotation.align_vectors(r_nrml, c_nrml)
    rotated = rot_tf.apply(coords - coords.mean(axis=0, keepdims=True))
    rotated = reference.std(axis=0, keepdims=True) * rotated / rotated.std(axis=0, keepdims=True)
    final = resolution * numpy.round(rotated / resolution).astype(int) + reference.mean(axis=0, keepdims=True)
    return final


def equalize_positions(coords, reference):
    coords = coords - coords.mean(axis=0, keepdims=True)
    coords = reference.std(axis=0, keepdims=True) * coords / coords.std(axis=0, keepdims=True)
    coords = coords + reference.mean(axis=0, keepdims=True)
    return coords


def discretize_positions(coords, resolution=100.):
    return resolution * numpy.round(coords / resolution).astype(int)


def homogenize_positions(coords):
    out = numpy.zeros_like(coords, dtype=float)
    perc = numpy.arange(0, 101)
    for i, c in enumerate(coords.transpose()):
        out[:, i] = numpy.interp(c, numpy.percentile(c, perc), perc)
    spread = coords.max(axis=0, keepdims=True) - coords.min(axis=0, keepdims=True)
    out = spread * out + coords.min(axis=0, keepdims=True)
    return out

def normalize_fm_coordinates(fm_coords, base_coords, normalization_args):
    if isinstance(normalization_args, list):
        for nrml in normalization_args:
            if nrml["method"] == "relax":
                fm_coords = relax_positions(fm_coords, base_coords, **nrml.get("kwargs", {}))
            elif nrml["method"] == "align":
                fm_coords = align_positions(fm_coords, base_coords, **nrml.get("kwargs", {}))
            elif nrml["method"] == "equalize":
                fm_coords = equalize_positions(fm_coords, base_coords)
            elif nrml["method"] == "discretize":
                fm_coords = discretize_positions(fm_coords, **nrml.get("kwargs", {}))
            elif nrml["method"] == "homogenize":
                fm_coords = homogenize_positions(fm_coords)
    else:
        return normalize_positions_manually(fm_coords, normalization_args)

def normalize_positions_manually(fm_coords, normalize_args=-1, multiply=numpy.NaN):
    fm_coords = fm_coords - numpy.nanmin(fm_coords, axis=0, keepdims=True)
    if fm_coords.shape[1] == 2:
        normalize_spread = numpy.array(normalize_args['normalize_spread'])
        offsets = numpy.array(normalize_args['normalize_offsets'])
    elif fm_coords.shape[1] > 2:
        normalize_spread = numpy.repeat(normalize_args['normalize_spread'][0], fm_coords.shape[1])
        offsets = numpy.repeat(normalize_args['normalize_offsets'][0], fm_coords.shape[1])
    multiply = numpy.array(multiply)
    if numpy.all(normalize_spread > 0):
        fm_coords = numpy.array(normalize_spread) * fm_coords / (numpy.nanmax(fm_coords, axis=0, keepdims=True) -
                                                                 numpy.nanmin(fm_coords, axis=0, keepdims=True))
        fm_coords = offsets + fm_coords
    if not numpy.any(numpy.isnan(multiply)):
        fm_coords = numpy.array(multiply) * fm_coords
    return fm_coords


if __name__ == "__main__":
    import argparse, os, json
    parser = argparse.ArgumentParser(description='Perform diffusion embedding of specified regions in a model of brain connectivity\n')

    parser.add_argument('--cache_config', help='Path to a json file specifying which projection cache class and parameters to use')
    parser.add_argument('--output_path', help='Flatmap output path', default='./db_flatmap.nrrd')
    parser.add_argument('--region_file', help='Json file that specifies which regions to flatten')
    parser.add_argument('--characterize', help='Set path to output file for the fraction variance explained by the selected components',
                        default=None)
    args = parser.parse_args()

    cache = projection_from_json(args.cache_config)
    brain_voxels_shape = cache._shape3d
    brain_voxel_sizes = None
    if hasattr(cache, "_voxel_sizes"):
        brain_voxel_sizes = cache._voxel_sizes

    output_path = args.output_path

    assert os.path.isfile(args.region_file), "Cannot find region file: {0}".format(str(args.region_file))
    with open(args.region_file, "r") as fid:
        regions = json.load(fid)

    if not isinstance(regions, list):
        regions = [regions]
    char = []
    for _regions in regions:
        tgt_options = _regions["connectivity_target"]
        components_to_use = _regions["components"]
        n_components = numpy.max(components_to_use) + 1
        if args.characterize is not None:
            print("Variance characterization requested. Setting number of components from {0} to 100".format(n_components))
            n_components = 100  # This is in the hope that 100 components are always enough to capture all variance... We test this later
        _, _, _, coords_final, embed_coords_final, embed_stats = flatten_pathway(cache,
                                                                       _regions["flatten"],
                                                                       tgt_options["considered_regions"],
                                                                       n_components=n_components,
                                                                       diffusion_time=1,
                                                                       direction=tgt_options["direction"])
        if args.characterize:
            if embed_stats["n_components_auto"] >= (embed_stats["n_components"] - 2):
                # If this is true then even the last component is stronger than 0.05 of the first.
                print("Warning: {0} components were apparently not enough to characterize the full variance. Resulting variance fractions not valid".format(embed_stats["n_components"]))
            component_lambda_ratios = embed_stats["lambdas"][components_to_use] / embed_stats["lambdas"].sum()

            char.append(dict({"region": _regions["flatten"][0], # adding lambdas for each region
                              "selected_lambda_ratios": list(component_lambda_ratios),
                              "lambdas": list(embed_stats["lambdas"])}))
                
        embed_coords_final = embed_coords_final[:, components_to_use]
        embed_coords_final = normalize_fm_coordinates(embed_coords_final, coords_final, _regions.get("normalization_args", {}))

        write_flatmap(_regions["flatten"], tgt_options["considered_regions"],
                      coords_final, embed_coords_final, output_path,
                      vxl_shape=brain_voxels_shape, vxl_dims=brain_voxel_sizes)
    if args.characterize: # saving diffusion results
        with open(args.characterize, "w") as fid:
            json.dump(char, fid)


