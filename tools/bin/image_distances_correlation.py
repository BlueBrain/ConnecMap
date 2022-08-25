'''
Compute various correlation coefficients for two distance series originating from two flat maps.

The common 3D source space is sampled to produce two distance series in the following way.
We first create a sample of pairs of voxels by sampling the 3D source space.
For each flat map, this sample of voxel pairs is used to create the series of the distances between
the corresponding voxels.
We compute then different correlation coefficients between these two image distance series.
'''
import os
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy
from scipy import signal
from skimage import morphology
from typing import Dict, List, Tuple
from nptyping import NDArray
from tqdm import tqdm
import logging
import itertools
import pandas as pd
from collections import defaultdict

from voxcell import VoxelData, RegionMap

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)

def compute_distance_map(
    flatmap: 'voxcell.VoxelData', voxel_indices: NDArray[int] = None, voxel_sample_size: int = 2000,
) -> Dict[int, List[float]]:
    """Compute a distance map corresponding to `flatmap`.

    This function returns a map which associates to each squared distance d^2 of between a pair of voxels in a sample of the source space
    the list of the distances between the voxel images in the target space for every voxel pair separated by d.

    Args:
        flatmap(voxcell.VoxelData): VoxelData object holding an integer array of shape (w, h, d, 2), to be
         interpreted as a map from a 3D volume to a 2D rectangle image.
        voxel_indices: (optional) voxel sample of the source region. Defaults to None.
        voxel_sample_size: (optional) size of the voxel sample of the flatmap source space.
         The number of computated values depends quadratically on `sample_size`.

    Returns:
        distance_map, i.e., a dict of the form
        {
            1: [1.0, 2.3, 5.0],
            4: [2.3, 5.0],
            9: [2.3],
            25: [1.0, 1.0, 5.0, 5.0],
            ...
        }
        The keys correspond to the squared distances between pairs of voxels taken in a sample of
        the source space and the values are lists of distances between the voxel images by the flat map.
    """
    assert issubclass(
        flatmap.raw.dtype.type, np.integer
    ), 'A flatmap must take integer values (pixel coordinates)'
    bounds = [np.max(flatmap.raw[..., i]) for i in range(2)]
    flatmap_raw = np.copy(flatmap.raw) / bounds[0] # Re-scale the flat map image width to be [0, 1] (x-axis)

    # Takes the indices of the voxels which have been mapped to some pixel.
    if voxel_indices is None:
        voxel_indices = np.array(np.where(flatmap_raw[..., 0] >= 0)).T
        step = voxel_indices.shape[0] // voxel_sample_size
        sample = np.arange(0, voxel_indices.shape[0], step=max(step, 1))
        voxel_indices = voxel_indices[sample]  # pylint: disable=unsubscriptable-object
    # All unordered pairs of distinct indices
    L.info('Computing index pairs ...')
    index_pairs = list(itertools.combinations(voxel_indices, 2))
    distance_map = defaultdict(dict)
    L.info('Gathering distance information in the distance map ...')
    for pair in tqdm(index_pairs):
        diff_source = np.array(pair[1]) - np.array(pair[0])
        squared_distance = np.dot(diff_source, diff_source)
        diff_target = flatmap_raw[tuple(pair[1])] - flatmap_raw[tuple(pair[0])]
        img_distance = np.linalg.norm(diff_target)
        if 'image_distances' not in distance_map[squared_distance]:
            distance_map[squared_distance]['image_distances'] = []
        distance_map[squared_distance]['image_distances'].append(img_distance)
        distance_map[squared_distance]['source_distance'] = np.linalg.norm(
            diff_source * flatmap.voxel_dimensions
        )

    return distance_map


def compute_local_distance_map(
    flatmap: 'voxcell.VoxelData', ball_sample: NDArray[bool], ball_center: NDArray[int]
) -> Dict[int, List[float]]:
    """Compute a local distance map corresponding to `flatmap`.

    This function returns a map which associates to each squared distance d^2 (d <= `ball_sample` radius)
    the list of the distances between the image of `ball_center` and the images of the voxels lying
    on the sphere of radius d centered at `ball_center` in the source space.

    Args:
        flatmap(voxcell.VoxelData): VoxelData object holding an integer array of shape (w, h, d, 2), to be
            interpreted as a map from a 3D volume to a 2D rectangle image.
        ball_sample: binary mask of a 3D ball lying inside the volume of interest.
        ball_center: 1D array holding the indices of the voxelcenter of `ball_sample`.

    Returns:
        distance_map, i.e., a dict of the form
        {
            1: [1.0, 2.3, 5.0],
            4: [2.3, 5.0],
            9: [2.3],
            25: [1.0, 1.0, 5.0, 5.0],
            ...
        }
        The keys correspond to the squared radii of spheres centered at `ball_center` and contained in `ball_sample`
        and the values are lists of distances between the image of `ball_center` and the images of the voxels lying on the
        sphere by the flat map.
    """
    assert issubclass(
        flatmap.raw.dtype.type, np.integer
    ), 'A flatmap must take integer values (pixel coordinates)'
    bounds = [np.max(flatmap.raw[..., i]) for i in range(2)]
    flatmap_raw = np.copy(flatmap.raw) / bounds[0]

    voxel_indices = np.array(np.where(ball_sample)).T
    center_mask = np.all(voxel_indices == ball_center, axis=1)
    center_index = np.where(center_mask)[0][0]
    voxel_indices = list(voxel_indices)
    del voxel_indices[center_index]

    distance_map = defaultdict(dict)
    L.info('Gathering distance information in the distance map ...')
    for index in tqdm(voxel_indices):
        diff_source = np.array(index) - np.array(ball_center)
        squared_distance = np.dot(diff_source, diff_source)
        if squared_distance > 0:
            diff_target = flatmap_raw[tuple(index)] - flatmap_raw[tuple(ball_center)]
            img_distance = np.linalg.norm(diff_target)
            if 'image_distances' not in distance_map[squared_distance]:
                distance_map[squared_distance]['image_distances'] = []
            distance_map[squared_distance]['image_distances'].append(img_distance)
            distance_map[squared_distance]['source_distance'] = np.linalg.norm(
                diff_source * flatmap.voxel_dimensions
            )

    return distance_map


def compute_metric_distorsion(distance_map: Dict[int, List[float]]) -> 'pd.DataFrame':
    """Compute the flatmap metric distorsion.

    This function returns a pandas.DataFrame holding the maximal, minimal and mean metric distorsion
    for every pair of voxels in a sample of the source space of the flatmap.

    The unit length is a voxel edge in the source space and a pixel edge in the target space.

    Args:
       distance_map(dict): dict of the form
       {
           1: [1.0, 2.3, 5.0],
           4: [2.3, 5.0],
           9: [2.3],
           25: [1.0, 1.0, 5.0, 5.0]
       }
       The keys correspond to the distances between pairs of voxels in the source space and the values are list of
       distances between the voxel images by the flat map.

    Returns:
        A pandas.DataFrame of the following form:
                 source_distance  max_image_distance  ...  mean_image_distance    stddev  metric_distorsion
        340         5.196152            2.236068  ...             2.236068          0.0        1.3334
        251        11.916375           36.687873  ...            36.687873          0.0        2.0999
        415        13.601471           18.000000  ...            18.000000          1.0        1.0001
        434        15.811388           25.019992  ...            25.019992          0.1        5.0667
        ...
        The `source_distance` column holds distances d between pairs of voxels sorted in increasing order.
        The `max_image_distance` column holds max || flat(v_1) - flat(v_2) || over all pairs of
        sampled voxels satisfying || v_1 - v_2 || = d.
        The `min_image_distance` column holds min || flat(v_1) - flat(v_2) || over all pairs of
        sampled voxels satisfying || v_1 - v_2 || = d.
        The `mean_image_distance` column holds mean || flat(v_1) - flat(v_2) || over all pairs of
        sampled voxels satisfying || v_1 - v_2 || = d.
        The `stddev` column holds the standard deviation of the distances || flat(v_1) - flat(v_2) || over all pairs of
        sampled voxels satisfying || v_1 - v_2 || = d.
        The `metric_distorsion` column holds the  max || flat(v_1) - flat(v_2) || / d over all pairs of
        sampled voxels satisfying || v_1 - v_2 || = d.
    """

    L.info('Populating the distance distorsion dataframe ...')
    data = defaultdict(list)
    for distances in tqdm(distance_map.values()):
        img_distances = distances['image_distances']
        data['source_distance'].append(distances['source_distance'])
        data['source_distance_ties'].append(len(img_distances))
        data['max_image_distance'].append(np.max(img_distances))
        data['min_image_distance'].append(np.min(img_distances))
        data['mean_image_distance'].append(np.mean(img_distances))
        data['stddev'].append(np.std(img_distances))

    result = pd.DataFrame(data=data)
    result.sort_values(by=['source_distance'], inplace=True)
    result['metric_distorsion'] = (
        result['max_image_distance'] / result['source_distance']
    )

    return result


def create_statistics_series(
    distance_map: Dict[int, List[float]],
) -> Tuple[NDArray[float], NDArray[float]]:
    """Gather source and target distance statistics under the form of two 1D series X and Y.

    Args:
       distance_map(dict): a dict as defined by the returned value of compute_distance_map.

    Returns:
        A tuple (x, y) where x and y and two lists of float of the same length. The list x is a
        series of distance between pairs of voxels taken in the source space of the flat map while
        y is the corresponding list of distances in the target space.
    """

    x = []
    y = []
    for distances in distance_map.values():
        img_distances = distances['image_distances']
        x += [distances['source_distance']] * len(img_distances)
        y += img_distances

    return x, y


def create_statistics(x: NDArray[float], y: NDArray[float]):
    """
    Create distance-related statistics

    Args:
        x(numpy.ndarray): 1D float array
        y(numpy.ndarray): 1D flat array with the same dimension as x.

    Returns:
        A dict containing the classical correlation coefficients of Pearson, Spearman and Kendall
        as well as the p values of the test for which the null hypothesis is that the coefficient is 0.
    """
    (
        slope,
        intercept,
        linear_correlation,
        linear_p_value,
        stderr,
    ) = scipy.stats.linregress(x, y)
    spearman_correlation, spearman_p_value = scipy.stats.spearmanr(x, y)
    kendall_correlation, kendall_p_value = scipy.stats.kendalltau(x, y)

    return {
        'linear_regression': {
            'slope': slope,
            'intercept': intercept,
            'linear_correlation_coefficient': linear_correlation,
            'p_value': linear_p_value,
            'standard_error': stderr,
        },
        'spearman_rho': {
            'spearman_correlation_coefficient': spearman_correlation,
            'p_value': spearman_p_value,
        },
        'kendall_tau': {
            'kendall_correlation_coefficient': kendall_correlation,
            'p_value': kendall_p_value,
        },
    }


def create_statistics_report(distance_map_1, distance_map_2):
    """
    Create a statistics report containing the correlation coefficients for each image distance variable.

    Args:
        distance_map_1: distance_map(dict): a dict as defined by the returned value of compute_distance_map.
        distance_map_2: distance_map(dict): a dict as defined by the returned value of compute_distance_map.

    Returns:
        a dict of the following form:
        image_distances: {
                'linear_regression': {
                    ...
                },
                'spearman_rho': {
                    ...
                },
                'kendall_tau': {
                    ...
                },
        }
    """

    report = {}
    _, x = create_statistics_series(distance_map_1)
    _, y = create_statistics_series(distance_map_2)
    report['image_distances'] = create_statistics(x, y)

    return report


def get_voxel_indices(
    flatmpap_1: 'VoxelData',
    flatmap_2: 'VoxelData',
    voxel_sample_size: int = 2000) -> NDArray[int]:
    """
    Args:
        flatmap_1: The first flatmap, i.e., a VoxelDataObject holding an integer array of shape (w, h, d, 2).
        flatmap_2: The second flatmap, i.e., a VoxelDataObject holding an integer array of shape (w, h, d, 2).
        voxel_sample_size: The number of voxels to be considered when sampling the underlying volume of
            `flatmap_1` and `flatmap_2`.

    Returns:
        voxel indices to be used for sampling the common source 3D region; tuple (X, Y, Z) where
        each coordinate holds an integer array of shape (voxel_sample_size,).
    """
    mask = np.logical_and(flatmap_1.raw[..., 0] >= 0, flatmap_2.raw[..., 0] >= 0)
    voxel_indices = np.array(np.where(mask)).T
    step = voxel_indices.shape[0] // voxel_sample_size
    sample = np.arange(0, voxel_indices.shape[0], step=max(step, 1))
    return voxel_indices[sample]  # pylint: disable=unsubscriptable-object



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute correlation between the image distances of two flat maps. \n'
        'This script takes two flat map nrrd files as input. '
        'The common 3D source space is sampled to produce two distance series in the following way.'
        'We first create a sample of pairs of voxels from the 3D source space.'
        'For each flat map, this sample of voxel pairs is used to create the series of'
        ' the distances between the corresponding voxels.'
        'We compute then different correlation coefficients between these two image distance series.'
    )
    parser.add_argument('flatmap_1_path', help='Path to the first flatmap nrrd file')
    parser.add_argument('flatmap_2_path', help='Path to the second flatmap nrrd file')
    parser.add_argument(
        'output_path',
        type=str,
        help='Path to the json file where to save the correlation coefficients',
    )
    parser.add_argument(
        '--voxel_sample_size',
        type=int,
        help='Size of the voxel sample used for metric computations. '
        'Defaults to None. '
        'The number of computed values depends quadratically on the sample size.',
        default=2000,
    )
    args = parser.parse_args()

    L.info('Loading flatmaps ...')
    flatmap_1 = VoxelData.load_nrrd(args.flatmap_1_path)
    flatmap_2 = VoxelData.load_nrrd(args.flatmap_2_path)

    L.info('Computing image distances ...')
    voxel_indices = get_voxel_indices(flatmap_1, flatmap_2, args.voxel_sample_size)
    distance_map_1 = compute_distance_map(flatmap_1, voxel_indices=voxel_indices)
    distance_map_2 = compute_distance_map(flatmap_2, voxel_indices=voxel_indices)

    L.info('Creating the statistics report ...')
    report = create_statistics_report(distance_map_1, distance_map_2)
    with open(
        Path(args.output_path), 'w', encoding='utf-8'
    ) as report_file:
        json.dump(report, report_file, separators=(',', ':'), indent=4)
