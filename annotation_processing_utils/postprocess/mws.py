# %%
import logging

logger: logging.Logger = logging.getLogger(name=__name__)

from cellmap_analyze.process.mutex_watershed import MutexWatershed


def mws(
    affinities_path,
    output_path,
    minimum_volume_nm_3,
    maximum_volume_nm_3,
    adjacent_edge_bias,
    lr_biases,
    filter_val,
    mask_config=None,
):
    mutex_watershed_processor = MutexWatershed(
        affinities_path=affinities_path,
        output_path=output_path,
        adjacent_edge_bias=adjacent_edge_bias,
        lr_bias_ratio=lr_biases,
        filter_val=filter_val,
        neighborhood=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [9, 0, 0],
            [0, 9, 0],
            [0, 0, 9],
        ],
        roi=None,
        minimum_volume_nm_3=minimum_volume_nm_3,
        maximum_volume_nm_3=maximum_volume_nm_3,
        mask_config=mask_config,
        num_workers=1,
        connectivity=3,
        padding_voxels=27,
        do_opening=True,
        delete_tmp=True,
        chunk_shape=None,
    )
    mutex_watershed_processor.get_connected_components()
