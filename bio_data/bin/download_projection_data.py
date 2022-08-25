#!/usr/bin/env python
import json


def main():
    import sys
    if len(sys.argv) < 2:
        print("""Usage: {0} path_to_config.json""".format(__file__))
        sys.exit(2)
    with open(sys.argv[1], "r") as fid:
        cfg = json.load(fid)

    from projection_voxels import CachedProjections
    import mcmodels
    cache = mcmodels.core.VoxelModelCache(manifest_file=cfg["Allen cache"])
    p = CachedProjections(allen_data=cache, cache_file=cfg["Hdf5 cache"])
    for src in cfg["Source regions"]:
        for tgt in cfg["Target regions"]:
            print("Trying to download data for {0} to {1}...".format(src, tgt))
            p.projection([src], [tgt])
            print("\t...Done!")


if __name__ == "__main__":
    main()
