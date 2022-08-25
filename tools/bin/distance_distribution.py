'''
Compute and render the histogram of the distances between pixel pairs in the flatmap image.

'''
from pathlib import Path
import numpy as np
import plotly.express as px
from typing import Dict, List, Tuple
from nptyping import NDArray
from tqdm import tqdm
import logging
import itertools


from voxcell import VoxelData, RegionMap

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)



def compute_flatmap_image(flatmap_raw: NDArray[int]) -> NDArray[bool]:
    """ Compute the binary image of the flatmap.

    This function returns a 2D boolean array representing the flatmap image.
    A pixel is white (True) if it is the image of a voxel by the
    flatmap. Otherwise the pixel is black (False).

    Args:
       flatmap_raw(numpy.ndarray): integer array of shape
            (l, w, h, 2), to be interpreted as a map from a 3D volume
            to a 2D rectangle image.
    Returns:
        boolean numpy.ndarray of shape (W, H) where W and H are
        the maximal values, augmented by 1, of the flatmap with respect to its last axis.
    """
    assert issubclass(flatmap_raw.dtype.type, np.integer)
    in_mask = np.all(flatmap_raw >= 0, axis=-1)
    pixel_indices = flatmap_raw[in_mask].T
    image_shape = np.max(pixel_indices, axis=1) + 1
    image = np.zeros(image_shape, dtype=np.bool)
    image[tuple(pixel_indices)] = True

    return image


def compute_pixel_pair_distances(flatmap_image: NDArray[bool], sample_size: int = None):
    # Takes the indices of the coloured pixels.
    pixel_indices = np.array(np.nonzero(flatmap_image)).T
    step = (
        pixel_indices.shape[0] // sample_size
        if sample_size is not None
        else 1
    )
    sample = np.arange(0, pixel_indices.shape[0], step=max(step, 1))
    pixel_indices = pixel_indices[sample]  # pylint: disable=unsubscriptable-object
    # All unordered pairs of distinct indices
    distances = []
    index_pairs = list(itertools.combinations(pixel_indices, 2))
    for pair in tqdm(index_pairs):
        diff = (
            np.array(pair[1]) - np.array(pair[0])
        )
        distances.append(np.linalg.norm(diff))

    return distances


def compute_distance_histogram(distances: List[float], max_distance: float, bins_count: int = None) -> Tuple[NDArray[float], NDArray[float]]:
    range = (0.0, max_distance)
    if bins_count is None:
        return np.histogram(distances, bins='auto', range=range)

    return np.histogram(distances, bins=bins_count, range=range)


def compute_distance_histogram_figure(counts:NDArray[float], bins: NDArray[float]) -> 'plotly.figure':
    bins = 0.5 * (bins[:-1] + bins[1:])
    return px.bar(x=bins, y=counts, labels={'x':'distance', 'y':'count'})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute and show the distance distribution of pixel pairs \n'
            ' in the flat map image.'
    )
    parser.add_argument('flatmap_path', help='Path to the flatmap nrrd file')
    parser.add_argument(
        'output_path',
        type=str,
        help='Path where to save the distance histogram image',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        help='Size of the pixel sample used for metric computations. '
        'Defaults to 2500. '
        'The number of computed values depends quadratically on the sample size.',
        default=2500,
    )
    parser.add_argument(
        '-s',
        '--show',
        action='store_true',
        help='Show the distance histogram.',
    )
    args = parser.parse_args()

    L.info('Loading flatmap ...')
    flatmap = VoxelData.load_nrrd(args.flatmap_path)
    L.info('Computing flatmap image ...')
    flatmap_image = compute_flatmap_image(flatmap.raw)
    max_distance = np.linalg.norm(flatmap_image.shape)
    L.info('Computing distances between pixels of the flatmap image ...')
    distances = compute_pixel_pair_distances(flatmap_image, sample_size=args.sample_size)
    counts, bins = compute_distance_histogram(distances, max_distance)
    L.info('Computing the distance histogram of the flatmap image ...')
    fig = compute_distance_histogram_figure(counts, bins)

    output_path = str(args.output_path)
    L.info('Saving histogram image to file ...')
    fig.write_image(output_path)
    if args.show:
        fig.show()
