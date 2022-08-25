'''
Compute and render metric information on a flatmap.

In this version, it is possible to query a subregion of the input annotated volume
and to carry a metric analysis on this subregion only.
See help documentation of the main() function.

The main function takes a flatmap nrrd file as input and save a png image file.
The image illustrates the flatmap metric distorsion by drawing three function graphs.
The represented functions are the functions which map a distance d > 0 to
- (1) max || flat(v_1) - flat(v_2) || over all pairs of voxels satisfying || v_1 - v_2 || = d,
- (2) min || flat(v_1) - flat(v_2) ||, idem,
- (3) mean || flat(v_1) - flat(v_2) ||, idem.
- (4) max || flat(v_1) - flat(v_2) || / d, idem.

The number || v || denotes either the Euclidean norm of a vector in R^3 or in R^2 depending on
whether v belongs to the source space or the target space of the flatmap `flat`.

The unit length in the source space is determined by the `voxel_dimensions` of the input `voxcell.VoxelData` flatmap.
The width of the flat map target space is re-scaled to [0, 1] while the aspect ratio is preserved. The flatmap target space
is equipped with the 2D Euclidean distance.
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
    flatmap: 'voxcell.VoxelData', volume: NDArray[bool], voxel_sample_size: int = 2000,
) -> Dict[int, List[float]]:
    """Compute a distance map corresponding to `flatmap`.

    This function returns a map which associates to each squared distance d^2 of between a pair of voxels in a sample of the source space
    the list of the distances between the voxel images in the target space for every voxel pair separated by d.

    Args:
        flatmap(voxcell.VoxelData): VoxelData object holding an integer array of shape (w, h, d, 2), to be
            interpreted as a map from a 3D volume to a 2D rectangle image.
        volume: 3D binary mask of the volume of interest.
        voxel_sample_size: size of the voxel sample of the flatmap source space.
            The number of computated values depends quadratically on `sample_size`. Defaults to 2000.

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
    flatmap_raw = np.copy(flatmap.raw) / bounds[0] # normalize the width of the flat map image

    # Takes the indices of the voxels in the volume of interest
    voxel_indices = np.array(np.where(volume)).T
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

    L.info('Grouping source and target distances as two satistical 1D series ...')
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


def create_statistics_report(distance_map, metric_data):
    """
    Create a statistics report containing the correlation coefficients for each image distance variable.

    Args:
        distance_map: distance_map(dict): a dict as defined by the returned value of compute_distance_map.

    Returns:
        a dict of the following form:
            {
                'distance': <dict>,
                'max_distance': <dict>,
                'mean_distance': <dict>
            }
        where each dict value is under the form defined by the returned value of create_statistics, i.e.,
        {
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
    x, y = create_statistics_series(distance_map)
    report['image_distance'] = create_statistics(x, y)

    source_distance = metric_data['source_distance'].to_numpy()
    report['max_image_distance'] = create_statistics(
        source_distance, metric_data['max_image_distance'].to_numpy()
    )
    report['mean_image_distance'] = create_statistics(
        source_distance, metric_data['mean_image_distance'].to_numpy()
    )
    report['min_image_distance'] = create_statistics(
        source_distance, metric_data['min_image_distance'].to_numpy()
    )
    report['metric_distorsion'] = create_statistics(
        source_distance, metric_data['metric_distorsion'].to_numpy()
    )

    return report


def compute_lipschitz_constant(
    flatmap_metric_data: 'pd.DataFrame',
) -> Tuple[float, float]:
    """
    Compute the Lipschitz constant of a sampled flatmap.

    The unit length is a voxel edge in the source space and a pixel edge in the target space.

    Args:
        flatmap_metric_data: data frame of the form described in the compute_metric_distorsion function.

    Returns:
        a tuple (lipsichitz_constant, optimum) where
            - `lipschitz_constant` is a float representing the maximum metric distorsion of a sampled flatmap.
            This number is max || flat(v_1) - flat(v_2) || / || v_1 - v_2 || taken over all the pairs of distinct sampled voxels.
            - `optimum` is the smallest distance between two voxels for which the above maximum is reached.
    """
    ratios = flatmap_metric_data['metric_distorsion']
    lipschitz_constant = np.max(ratios)
    optimum = flatmap_metric_data['source_distance'][ratios.argmax()]
    return lipschitz_constant, optimum


def save_metric_info_graphs(
    metric_data: 'pd.DataFrame',
    output_path: str,
    distance_sample_size: int = 8000,
    show: bool = True,
) -> None:
    """
    Show in the web browser 3 function graphs illustrating the flatmap metric distorsion.

    The three represented functions are the functions which map a distance d > 0 to:
    - max || flat(v_1) - flat(v_2) || over all pairs of sampled voxels satisfying || v_1 - v_2 || = d,
    - min || flat(v_1) - flat(v_2) || idem,
    - mean || flat(v_1) - flat(v_2) || idem.
    - max || flat(v_1) - flat(v_2) || / d, idem.

    The image is saved to file only if `output_path` is specified.

    Args:
        metric_data: data frame of the form described in compute_metric_distorsion.
        output_path: a string indicating where to save the output image.
        distance_sample_size: number of source distance values used to plot the 4 graphs.
        show: If True, show the metric distorsion graphs in the browser
    """
    x = metric_data['source_distance']
    # Down sampling
    step = x.shape[0] // distance_sample_size if distance_sample_size is not None else 1
    sample = np.arange(0, x.shape[0], step=max(step, 1))
    x = x.to_numpy()[sample]
    filter_window_size = (len(x) // 20) | 1  # must be odd

    fig = make_subplots(rows=2, cols=1)

    for label, color in zip(
        ['max_image_distance', 'mean_image_distance', 'min_image_distance'],
        ['red', 'green', 'blue'],
    ):
        y = metric_data[label].to_numpy()[sample]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                opacity=0.3,
                marker=go.scatter.Marker(
                    size=4,
                    color=color,
                ),
                name=label,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=signal.savgol_filter(
                    y,
                    filter_window_size,  # window size used for filtering, must be odd
                    3,
                ),  # order of fitted polynomial
                mode='lines',
                line=go.scatter.Line(
                    color=color,
                ),
                name=label + ' (Savitzky-Golay)',
            ),
            row=1,
            col=1,
        )

    y = metric_data['metric_distorsion'].to_numpy()[sample]
    #y = np.maximum.accumulate(y) # optimal continuity modulus

    fig.add_trace(
        go.Scatter(
            x=x, # normalize distances in the source space
            y=y,
            mode="markers",
            opacity=0.7,
            marker=go.scatter.Marker(
                size=4,
                color='purple',
            ),
            name='Metric distorsion',
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=signal.savgol_filter(
                y,
                filter_window_size,  # window size used for filtering, must be odd
                3,
            ),  # order of fitted polynomial
            mode='lines',
            line=go.scatter.Line(
                color='purple',
            ),
            name='Metric distorsion (Savitzky-Golay)',
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=1500, width=1500, title_text="Flat Map Metric Distorsion Graphs"
    )

    output_path = str(output_path)
    fig.write_image(output_path)

    if show:
        fig.show()


def save_boxes_distorsion(
    flatmap: 'voxcell.VoxelData',
    output_path: str,
    grid_size: int = 10,
    pixel_sample_size: int = 3000,
    show: bool = True,
) -> None:
    """
    Show how boxes in the input volume are distorted in the flat map image.

    A png image is saved to disk at the `output_path` location.

    Args:
        flatmap: VoxelData object holding an integer array of shape (w, h, d, 2), to be
            interpreted as a map from a 3D volume to a 2D rectangle image.
        output_path: Filepath of the png image to be saved.
        grid_size: size of the 3D grid from which boxes are extracted.
            The number of boxes is less or equal to `grid_size` to the cube.
        pixel_sample_size: maximal number of pixels used to represent the image of a box.
        show: if True, the saved image is displayed into the web browser.
    """
    fig = go.Figure()
    shape = flatmap.raw.shape[:-1]
    sizes = np.array(shape) // grid_size
    ranges = [range(0, shape[i], sizes[i]) for i in range(3)]
    indices = list(itertools.product(*ranges))

    L.info('Creating figure for box distorsion ...')
    for id_, center in tqdm(enumerate(indices)):
        bottom = np.max([np.array(center) - sizes // 3, [0, 0, 0]], axis=0)
        top = np.min([np.array(center) + sizes // 3, shape], axis=0)
        slices = tuple(slice(bottom[i], top[i]) for i in range(3))
        box = flatmap.raw[slices]
        x, y = box[..., 0], box[..., 1]
        x = x[x >= 0]
        y = y[y >= 0]
        if len(x) == 0:  # Empty intersection with the source volume
            continue

        # Downsample for performance reason
        x = x[slice(0, len(x), max(len(x) // pixel_sample_size, 1))]
        y = y[slice(0, len(y), max(len(y) // pixel_sample_size, 1))]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                opacity=0.8,
                marker=go.scatter.Marker(
                    size=4,
                ),
                name=f'Box {id_}',
            ),
        )

    output_path = str(output_path)
    fig.write_image(output_path)

    if show:
        L.info('Show box distorsion in the browser ...')
        fig.show()


def get_ball(volume_mask: NDArray[bool], center: NDArray[int], radius: int):
    ball = morphology.ball(radius, dtype=np.bool)
    ball_mask = np.zeros_like(volume_mask)
    center = np.array(center)
    bounding_box = np.array([center - radius, center + radius])
    bounding_box[0] = np.max([bounding_box[0], [0, 0, 0]], axis=0)
    bounding_box[1] = np.min([bounding_box[1], np.array(volume_mask.shape)], axis=0)
    ball_bounds = [bounding_box[0] - center + radius, bounding_box[1] - center + radius]
    ball_mask[
        bounding_box[0][0] : bounding_box[1][0],
        bounding_box[0][1] : bounding_box[1][1],
        bounding_box[0][2] : bounding_box[1][2],
    ] = ball[
        ball_bounds[0][0] : ball_bounds[1][0],
        ball_bounds[0][1] : ball_bounds[1][1],
        ball_bounds[0][2] : ball_bounds[1][2]
    ]

    return np.logical_and(ball_mask, volume_mask)


def split_str(value_str, new_type, sep=','):
    return list(map(new_type, value_str.strip().split(sep)))


def string_to_type_converter(string):
    """ Convert a string to the type it refers to.

    Args:
        string(str): string corresponding to a python
        basic data type, i.e., 'int', 'float', 'str' or
        'bool'.

    Returns:
        The python type referred by the input string.
    """
    CONVERTER = {'int': int, 'float': float, 'str': str, 'bool': bool}
    return CONVERTER[string]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute and show information on flatmap metric distorsion. \n'
        'This script takes a flatmap nrrd file as input, shows and optionally saves a png image file. '
        'The image illustrates the flatmap metric distorsion by drawing 4 function graphs. '
        'The represented functions are the functions which map a distance d > 0 to \n'
        ' - max || flat(v_1) - flat(v_2) || over all pairs of voxels satisfying || v_1 - v_2 || = d,\n'
        ' - min || flat(v_1) - flat(v_2) ||, idem, \n'
        ' - mean || flat(v_1) - flat(v_2) ||, idem. \n'
        ' - max || flat(v_1) - flat(v_2) || / d, idem. \n '
        'The number || v || denotes either the Euclidean norm of a vector in R^3 or in R^2 depending on '
        'whether v belongs to the source space or the target space of the flatmap `flat`.\n'
        'The unit length is defined by the `voxel_dimensions` of the input `voxcell.VoxelData` flatmap in the source space.'
        'The target space width is normalized to be [0, 1] and the aspect ratio is preserved. The target space is equipped'
        ' with the Euclidean distance.'
    )
    parser.add_argument('flatmap_path', help='Path to the flatmap nrrd file')
    parser.add_argument('brain_region_path', help='Path to the brain region nrrd file')
    parser.add_argument('hierarchy_path', help='Path to the brain region hierarchy file, e.g., 1.json or hierarchy.json')
    parser.add_argument(
        'output_dir',
        type=str,
        help='Folder path where to save the metric distorsion graphs and statistics report',
    )
    parser.add_argument(
        'field_name',
        type=str,
        help='Name of the field used to query a subregion of the input "brain_region", e.g,  "acronym" or "name"',
    )
    parser.add_argument(
        'value',
        type=str,
        help='Value of the field used to query a subregion of the input "brain_region", '
        'e.g, an uint32 id like 169, or an acronym like "Isocortex"',
    )
    parser.add_argument(
        'value_type',
        type=str,
        help='Type of the field value used to query a subregion of the input "brain_region", '
        'e.g, "int" if the field name is "id", or "str" if the field name is "Isocortex"',
    )
    parser.add_argument(
        '-d',
        '--with_descendants',
        help='Return all the descendant subregions of the hierarchy query',
        action='store_true',
        required=False,
        default=True
    )
    parser.add_argument(
        '-s',
        '--voxel_sample_size',
        type=int,
        help='Size of the voxel sample used for metric computations. '
        'Defaults to 2000. '
        'The number of computed values depends quadratically on the sample size.',
        required=False,
        default=2000,
    )

    args = parser.parse_args()
    L.info('Loading hierarchy json file ...')
    region_map = RegionMap.load_json(args.hierarchy_path)

    L.info('Loading annotated volume (nrrd) ...')
    annotated_volume = VoxelData.load_nrrd(args.brain_region_path)

    L.info('Loading flatmap (nrrd) ...')
    flatmap = VoxelData.load_nrrd(args.flatmap_path)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    value = string_to_type_converter(args.value_type)(args.value)
    ids = list(region_map.find(value, args.field_name, with_descendants=args.with_descendants))
    volume = np.isin(annotated_volume.raw, ids) & (flatmap.raw[..., 0] >= 0)

    #L.info('Saving the image of boxes distorsion ...')
    #save_boxes_distorsion(flatmap, Path(args.output_dir, 'boxes_distorsion.png'))

    L.info('Computing flatmap metric distorsion information ...')
    distance_map = compute_distance_map(flatmap, volume, voxel_sample_size=args.voxel_sample_size)

    metric_data = compute_metric_distorsion(distance_map)
    L.info('Saving flatmap metric distorsion graphs ...')
    save_metric_info_graphs(
        metric_data,
        Path(args.output_dir, 'metric_distorsion_graphs.png'),
        distance_sample_size=10000,
        show=True,
    )

    lipshitz_constant, optimum = compute_lipschitz_constant(metric_data)
    # L.info('The Lipschitz constant of this flatmap sample is %f. '
    #    'It was obtained for a source distance equal to %f.',
    #    lipshitz_constant, next(iter(optimum))
    # )

    L.info('Creating the statistics report ...')
    report = create_statistics_report(distance_map, metric_data)
    report['lipschitz_constant'] = {
        'constant': lipshitz_constant,
        'optimum': optimum,
    }
    report['stddev_average'] = np.mean(metric_data['stddev'])
    with open(
        Path(args.output_dir, 'statistics_report.json'), 'w', encoding='utf-8'
    ) as report_file:
        json.dump(report, report_file, separators=(',', ':'), indent=4)
