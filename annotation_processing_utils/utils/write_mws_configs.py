# %%
import logging
from annotation_processing_utils.utils.parse_data_path import parse_data_path
from datetime import datetime
import yaml
import getpass
import os

logger: logging.Logger = logging.getLogger(name=__name__)


def write_mws_configs(affinities_path: str, segmentation_path: str):
    _, affinities_dataset = parse_data_path(affinities_path)
    # Define dask configuration
    dask_config = {
        "jobqueue": {
            "lsf": {
                "ncpus": 8,
                "processes": 12,
                "cores": 12,
                "memory": "120GB",
                "walltime": "08:00",
                "mem": 12000000000,
                "use-stdin": True,
                "log-directory": "job-logs",
                "name": "cellmap-analyze",
                "project": "charge_group",
            }
        },
        "distributed": {
            "scheduler": {"work-stealing": True},
            "admin": {
                "log-format": "[%(asctime)s] %(levelname)s %(message)s",
                "tick": {"interval": "20ms", "limit": "3h"},
            },
        },
    }

    # Collect all arguments in a dictionary
    args = {
        "affinities_path": affinities_path,
        "output_path": segmentation_path,
        "connectivity": 3,
        "adjacent_edge_bias": -0.4,
        "lr_bias_ratio": -0.08,
        "filter_val": 0.5,
        "neighborhood": [
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
        "mask_config": None,
        "roi": None,
        "minimum_volume_nm_3": 45037,
        "maximum_volume_nm_3": "inf",
        "num_workers": 10,
        "padding_voxels": None,
        "do_opening": True,
        "delete_tmp": True,
        "chunk_shape": None,
        "dask_config": dask_config,
        "timestamp": datetime.now().isoformat(),
    }

    username = getpass.getuser()
    timestamp = datetime.now().strftime(f"%Y%m%d/%H%M%S/")
    log_dir = f"/nrs/cellmap/{username}/logs/cellmap_analyze_logs/mws_logs/{timestamp}/{affinities_dataset}_filter_val_{args['filter_val']}_lrb_ratio_{args['lr_bias_ratio']}_adjacent_edge_bias_{args['adjacent_edge_bias']}/"
    os.makedirs(log_dir, exist_ok=True)
    # Generate output YAML filename based on the segmentation path

    with open(f"{log_dir}/run-config.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file, default_flow_style=False, indent=2)

    with open(f"{log_dir}/dask-config.yaml", "w") as yaml_file:
        yaml.dump(dask_config, yaml_file, default_flow_style=False, indent=2)

    return log_dir


# %%
affinities_path = "/groups/cellmap/cellmap/ackermand/Programming/annotation-processing-utils/asdfasdf.zarr/blahasd"
segmentation_path = "/groups/cellmap/cellmap/ackermand/Programming/annotation-processing-utils/asdasf.zarr/bloop"
log = write_mws_configs(affinities_path, segmentation_path)
# %%
