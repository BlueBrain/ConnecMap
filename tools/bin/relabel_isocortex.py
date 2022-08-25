'''
Relabel the AIBS Mouse Isocortex using only the 43 AIBS StructureIDs of the main isocortex regions

Lexicon:
    * AIBS: Allen Institute for Brain Science

Assumptions:
    * The annotation nrrd file to be used is CCFv3 2017 Mouse Brain Atlas file to be found here
        http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/
    * We want to restrict attention to the 43 main isocortical regions listed below
'''
import numpy as np
from typing import Set
from nptyping import NDArray
from tqdm import tqdm
import logging

from voxcell import VoxelData, RegionMap

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)

# The 43 main isocortical regions of the mouse brain
ISOCORTICAL_REGIONS = [
    "FRP",
    "MOp",
    "MOs",
    "SSp-n",
    "SSp-bfd",
    "SSp-ll",
    "SSp-m",
    "SSp-ul",
    "SSp-tr",
    "SSp-un",
    "SSs",
    "GU",
    "VISC",
    "AUDd",
    "AUDp",
    "AUDpo",
    "AUDv",
    "VISal",
    "VISam",
    "VISl",
    "VISp",
    "VISpl",
    "VISpm",
    "VISli",
    "VISpor",
    "ACAd",
    "ACAv",
    "PL",
    "ILA",
    "ORBl",
    "ORBm",
    "ORBvl",
    "AId",
    "AIp",
    "AIv",
    "RSPagl",
    "RSPd",
    "RSPv",
    "VISa",
    "VISrl",
    "TEa",
    "PERI",
    "ECT",
]


def relabel_isocortex_main_regions(
    region_map: 'RegionMap',
    annotation_raw: NDArray[np.uint32],
    copy=False
    ) -> NDArray[np.uint32]:

    if copy:
        annotation_raw = annotation_raw.copy()

    # Uses a unique label of each of the 43 main isocortical regions
    main_isocortical_region_ids: Set[np.uint32] = set()
    for region in tqdm(ISOCORTICAL_REGIONS):
        ids = region_map.find(region, attr='acronym', with_descendants=True)
        mask = np.isin(annotation_raw, list(ids))
        id_ = next(iter(list(region_map.find(region, attr='acronym'))))
        annotation_raw[mask] = id_
        main_isocortical_region_ids = main_isocortical_region_ids | ids

    # Zeroes every region that is not listed in ISOCORTICAL_REGIONS
    out_mask = np.isin(annotation_raw, list(main_isocortical_region_ids), invert=True)
    annotation_raw[out_mask] = 0

    return annotation_raw


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Relabel the AIBS Mouse Isocortex using only the 43 AIBS StructureIDs of the main isocortex regions')
    parser.add_argument('hierarchy_path', help='Path to 1.json or hierarchy.json. For unmodified AIBS Mouse Brain Atlas, get'
        ' http://api.brain-map.org/api/v2/structure_graph_download/1.json')
    parser.add_argument('annotation_path', help='Path to the annotation nrrd file')
    parser.add_argument('output_path', help='Path where to save the re-labeled annotation file')
    args = parser.parse_args()

    region_map = RegionMap.load_json(args.hierarchy_path)
    L.info('Loading annotation ...')
    annotation = VoxelData.load_nrrd(args.annotation_path)

    L.info('Re-labelling the %d mouse isocortex main regions with their AIBS StructureIDs ...', len(ISOCORTICAL_REGIONS))
    annotation.raw = relabel_isocortex_main_regions(region_map, annotation.raw, copy=False)

    L.info('Saving the re-labeled annotation file to %s ...', args.output_path)
    annotation.save_nrrd(args.output_path)
