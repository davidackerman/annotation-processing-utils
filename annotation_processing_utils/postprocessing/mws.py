from pathlib import Path
import rusty_mws
import logging
from annotation_processing_utils.utils.parse_data_path import parse_data_path
from datetime import datetime

logger: logging.Logger = logging.getLogger(name=__name__)
import getpass


def mws(
    affinities_path: str,
    segmentation_path: str,
):
    affinities_file, affinities_dataset = parse_data_path(affinities_path)
    output_file, output_dataset = parse_data_path(segmentation_path)
    username = getpass.getuser()
    for adj_bias, lr_bias in [(0.5, -1.2)]:  # ,(0.1,-1.2)]:
        filter_val = 0.5
        lr_bias_ratio = -0.08
        timestamp = datetime.now().strftime(f"%Y%m%d/%H%M%S/")
        log_dir = f"/nrs/cellmap/{username}/logs/daisy_logs/mws_logs/{affinities_dataset}_filter_val_{filter_val}_lrb_ratio_{lr_bias_ratio}_adj_{adj_bias}_lr_{lr_bias}/{timestamp}"
        pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
            # sample_name="test",
            affs_file=affinities_file,
            affs_dataset=affinities_dataset,
            fragments_file=output_file,
            fragments_dataset=f"{output_dataset}_filter_val_{filter_val}_lrb_ratio_{lr_bias_ratio}_adj_{adj_bias}_lr_{lr_bias}_frags",
            seg_file=output_file,
            seg_dataset=f"{output_dataset}_filter_val_{filter_val}_lrb_ratio_{lr_bias_ratio}_adj_{adj_bias}_lr_{lr_bias}_segs",
            db_host="mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017",
            db_name=f"rusty_mws_{username}",
            lr_bias_ratio=lr_bias_ratio,
            adj_bias=adj_bias,
            lr_bias=lr_bias,
            nworkers_frags=44,
            nworkers_lut=44,
            nworkers_supervox=44,
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
            log_dir=log_dir,
        )
        success = pp.run_pred_segmentation_pipeline()
        if success:
            print(
                f"completed ({affinities_dataset}_filter_val_{filter_val}_lrb_ratio_{lr_bias_ratio}_adj_{adj_bias}_lr_{lr_bias}) all tasks successfully!"
            )
        else:
            print(
                f"failed ({affinities_dataset}_filter_val_{filter_val}_lrb_ratio_{lr_bias_ratio}_adj_{adj_bias}_lr_{lr_bias}) with status {success} Some task failed."
            )
