'''
Compute and render metric information on a flatmap.

The main function takes a flatmap nrrd file as input and save a png image file.
The image illustrates the flatmap metric distorsion by drawing three function graphs.
The represented functions are the functions which map a distance d > 0 to
- max || flat(v_1) - flat(v_2) || over all pairs of voxels satisfying || v_1 - v_2 || = d,
- min || flat(v_1) - flat(v_2) ||, idem,
- mean || flat(v_1) - flat(v_2) ||, idem.

The number || v || denotes either the Euclidean norm of a vector in R^3 or in R^2 depending on
whether v belongs to the source space or the target space of the flatmap `flat`.

The unit length is a voxel edge in the source space and a pixel edge in the target space.
'''

import numpy as np
import plotly.graph_objects as go
import scipy
from scipy import signal
from typing import List, Tuple
from nptyping import NDArray
from tqdm import tqdm
import logging
import itertools
import pandas as pd
from collections import defaultdict

from voxcell import VoxelData, RegionMap

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)



def compute_metric_distorsion(flatmap_raw: NDArray[int], voxel_sample_size: int = None,
) -> NDArray[int]:
    """ Compute the flatmap metric distorsion.

    This function returns a pandas.DataFrame holding the maximal, minimal and mean metric distorsion
    for every pair of voxels in a sample of the source space of the flatmap.

    The unit length is a voxel edge in the source space and a pixel edge in the target space.

    Args:
       flatmap_raw(numpy.ndarray): integer array of shape (w, h, d, 2), to be
         interpreted as a map from a 3D volume to a 2D rectangle image.
        sample_size: size of the voxel sample of the flatmap source space.
            If None, all the voxels of the source space are used. The number of computated values depends quadratically on
            `sample_size`. Defaults to None.

    Returns:
        A pandas.DataFrame of the following form:
                 source_distance  max_image_distance  ...  mean_image_distance  stddev
        340         5.196152            2.236068  ...             2.236068     0.0
        251        11.916375           36.687873  ...            36.687873     0.0
        415        13.601471           18.000000  ...            18.000000     0.0
        434        15.811388           25.019992  ...            25.019992     0.0
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
    """
    assert issubclass(flatmap_raw.dtype.type, np.integer)
    # Takes the indices of the voxels which have been mapped to some pixel.
    voxel_indices = np.array(np.where(flatmap_raw[..., 0] >= 0)).T
    step = voxel_indices.shape[0] // voxel_sample_size if voxel_sample_size is not None else 1
    sample = np.arange(0, voxel_indices.shape[0], step=max(step, 1))

    voxel_indices = voxel_indices[sample]  # pylint: disable=unsubscriptable-object
    # All unordered pairs of distinct indices
    index_pairs = list(itertools.combinations(voxel_indices, 2))
    distance_map = defaultdict(list)
    L.info('Collecting distance information ...')
    for pair in tqdm(index_pairs):
        diff = np.array(pair[1]) - np.array(pair[0])
        squared_distance = np.dot(diff, diff)
        diff = flatmap_raw[tuple(pair[1])] - flatmap_raw[tuple(pair[0])]
        img_distance = np.linalg.norm(diff)
        distance_map[squared_distance].append(img_distance)

    L.info('Populating the distance distorsion dataframe ...')
    data = defaultdict(list)
    for squared_distance, distances in distance_map.items():
        data['source_distance'].append(np.sqrt(squared_distance))
        data['max_image_distance'].append(np.max(distances))
        data['min_image_distance'].append(np.min(distances))
        data['mean_image_distance'].append(np.mean(distances))
        data['stddev'].append(np.std(distances))

    result = pd.DataFrame(data=data)
    result.sort_values(by=['source_distance'], inplace=True)

    return result

def compute_lipschitz_constant(flatmap_metric_data: 'pd.DataFrame') -> Tuple[float, float]:
    '''
    Compute the Lipschitz constant of a sampled flatmap.

    The unit length is a voxel edge in the source space and a pixel edge in the target space.

    Args:
        flatmap_metric_data: data frame of the form described in the compute_metric_distorsion function.

    Returns:
        a tuple (lipsichitz_constant, optimum) where
            - `lipschitz_constant` is a float representing the maximum metric distorsion of a sampled flatmap.
            This number is max || flat(v_1) - flat(v_2) || / || v_1 - v_2 || taken over all the pairs of distinct sampled voxels.
            - `optimum` is the smallest distance between two voxels for which the above maximum is reached.
    '''
    ratios = flatmap_metric_data['max_image_distance'] / metric_data['source_distance']
    lipschitz_constant = np.max(ratios)
    optimum = flatmap_metric_data['source_distance'][ratios.argmax()],
    return  lipschitz_constant, optimum

def show_metric_info_graphs(metric_data: 'pd.DataFrame', output_path: str = '', distance_sample_size=None) -> None:
    '''
    Show in the web browser 3 function graphs illustrating the flatmap metric distorsion.

    The three represented functions are the functions which map a distance d > 0 to:
    - max || flat(v_1) - flat(v_2) || over all pairs of sampled voxels satisfying || v_1 - v_2 || = d,
    - min || flat(v_1) - flat(v_2) || idem,
    - mean || flat(v_1) - flat(v_2) || idem.

    The image is saved to file only if `output_path` is specified.

    Args:
        metric_data: data frame of the form described in compute_metric_distorsion.
        output_path: either a string indicating where to save the output image of None, in which case,
            no image is saved. Defaults to None.
        distance_sample_size: number of source distance values used to plot the 3 graphs. Defaults to None, in which case
            all the source distance values of `metric_data` will be used.



    '''
    x = metric_data['source_distance']
    # Down sampling
    step = x.shape[0] // distance_sample_size if distance_sample_size is not None else 1
    sample = np.arange(0, x.shape[0], step=max(step, 1))
    x = x.to_numpy()[sample]
    filter_window_size = (len(x) // 20) | 1 # must be odd

    fig = go.Figure()
    for label, color in zip(
        ['max_image_distance', 'mean_image_distance', 'min_image_distance'],
        ['red', 'green', 'blue']):
        y = metric_data[label].to_numpy()[sample]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            opacity=0.3,
            marker=go.scatter.Marker(
                size=2,
                color=color,
            ),
            name=label,
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=signal.savgol_filter(y,
                                filter_window_size, # window size used for filtering, must be odd
                                3), # order of fitted polynomial
            mode='lines',
            line=go.scatter.Line(
                color=color,
            ),
            name=label + ' (Savitzky-Golay)'
        ))


    if output_path:
        fig.write_image(output_path)
    fig.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute and show information on flatmap metric distorsion. \n'
        'This script takes a flatmap nrrd file as input, show and optionally save a png image file. '
        'The image illustrates the flatmap metric distorsion by drawing three function graphs. '
        'The represented functions are the functions which map a distance d > 0 to '
        ' - max || flat(v_1) - flat(v_2) || over all pairs of voxels satisfying || v_1 - v_2 || = d,\n'
        ' - min || flat(v_1) - flat(v_2) ||, idem, \n'
        ' - mean || flat(v_1) - flat(v_2) ||, idem. \n'
        'The number || v || denotes either the Euclidean norm of a vector in R^3 or in R^2 depending on '
        'whether v belongs to the source space or the target space of the flatmap `flat`.\n'
        'The unit length is a voxel edge in the source space and a pixel edge in the target space.'
    )
    parser.add_argument('flatmap_path', help='Path to the flatmap nrrd file')
    parser.add_argument('--output_path', help='File path where to save the metric distorsion graphs', default=None)
    parser.add_argument('--voxel_sample_size', help='Size of the voxel sample used for metric computations. '
        'Defaults to 1000. '
        'The number of computed values depends quadratically on the sample size.', default=1000)
    args = parser.parse_args()

    L.info('Loading flatmap ...')
    flatmap = VoxelData.load_nrrd(args.flatmap_path)

    L.info('Computing flatmap metric distorsion information ...')
    metric_data = compute_metric_distorsion(flatmap.raw, voxel_sample_size=args.voxel_sample_size)

    L.info('Showing flatmap metric distorsion graphs ...')
    show_metric_info_graphs(metric_data, args.output_path, distance_sample_size=15000)

    lipshitz_constant, optimum = compute_lipschitz_constant(metric_data)
    L.info('The Lipschitz constant of this flatmap sample is %f. '
        'It was obtained for a source distance equal to %f.',
        lipshitz_constant, next(iter(optimum))
    )
