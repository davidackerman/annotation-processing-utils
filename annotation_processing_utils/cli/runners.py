from datetime import datetime
from html import parser
import os
from annotation_processing_utils.utils.parse_data_path import parse_data_path
from annotation_processing_utils.utils.write_mws_configs import write_mws_configs
import argparse
import logging
import getpass
import json
import ast

logger: logging.Logger = logging.getLogger(name=__name__)


def run_dacapo_train():
    from dacapo.store.create_store import create_config_store
    from dacapo.train import train_run
    from dacapo.experiments.run import Run

    parser = argparse.ArgumentParser(description="Run training via DaCapo")
    parser.add_argument("--run", type=str, help="Name of the run", required=True)
    parser.add_argument(
        "--do_validate",
        type=bool,
        default=False,
        help="Whether to do validation",
        required=False,
    )
    args = parser.parse_args()

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(args.run)
    run = Run(run_config)
    train_run(run, args.do_validate)


def run_inference():
    from annotation_processing_utils.postprocess.inference import inference
    from funlib.geometry import Roi

    parser = argparse.ArgumentParser(
        description="Run inference for a particular run, raw dataset and roi"
    )
    parser.add_argument("--run", type=str, help="The run name", required=True)
    parser.add_argument("--iteration", type=int, help="Iteration", required=True)
    parser.add_argument(
        "--raw_path", type=str, help="Path to the raw data", required=True
    )
    parser.add_argument(
        "--inference_path",
        type=str,
        help="The output path for the prediction",
        required=True,
    )
    parser.add_argument(
        "--roi_offset",
        type=str,
        help="The offset of the roi - in world units - as a comma separated list",
        required=True,
    )
    parser.add_argument(
        "--roi_shape",
        type=str,
        help="The shape of the roi - in world units -  as a comma separated list",
        required=True,
    )

    args = parser.parse_args()

    roi = Roi(
        [float(c) for c in args.roi_offset.split(",")],
        [float(c) for c in args.roi_shape.split(",")],
    )
    inference(args.run, args.iteration, args.raw_path, args.inference_path, roi)


def run_mws():

    from annotation_processing_utils.postprocess.mws import mws
    import numpy as np

    parser = argparse.ArgumentParser(
        description="Run mutex watershed segmentation for an affinities dataset"
    )
    parser.add_argument(
        "--affinities_path", type=str, help="Path to the affinities", required=True
    )
    parser.add_argument(
        "--segmentation_path",
        type=str,
        help="Output path for segmentations",
        required=True,
    )
    parser.add_argument(
        "--minimum_volume_nm_3",
        type=int,
        help="Minimum volume in nm^3",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--maximum_volume_nm_3",
        type=float,
        help="Maximum volume in nm^3",
        required=False,
        default=np.inf,
    )
    parser.add_argument(
        "--adjacent_edge_bias",
        type=float,
        help="Bias for adjacent edges",
        required=False,
        default=-0.4,
    )
    parser.add_argument(
        "--lr_biases",
        type=float,
        nargs="+",  # one or more numbers
        metavar="F",
        help="Bias(es) for long-range edges (one or more floats).",
        default=[-0.24, -0.72],
    )

    parser.add_argument(
        "--filter_val",
        type=float,
        help="Filter value for merged object affinities",
        required=False,
        default=0.5,
    )

    parser.add_argument(
        "--mask_config",
        default=None,
        type=str,
        help="Mask if applicable (where a 0 indicates a region to ignore)",
        required=False,
    )
    args = parser.parse_args()
    mask_config = args.mask_config
    if mask_config is not None:
        mask_config = json.loads(mask_config)
    mws(
        args.affinities_path,
        args.segmentation_path,
        args.minimum_volume_nm_3,
        args.maximum_volume_nm_3,
        args.adjacent_edge_bias,
        args.lr_biases,
        args.filter_val,
        mask_config,
    )


def run_rusty_mws_processor():
    from annotation_processing_utils.postprocess.rusty_mws_processor import (
        rusty_mws_processor,
    )

    parser = argparse.ArgumentParser(
        description="Run rusty mutex watershed segmentation for an affinities dataset"
    )
    parser.add_argument(
        "--affinities_path", type=str, help="Path to the affinities", required=True
    )
    parser.add_argument(
        "--segmentation_path",
        type=str,
        help="Output path for segmentations",
        required=True,
    )

    parser.add_argument(
        "--mask_path",
        default=None,
        type=str,
        help="Mask if applicable (where a 0 indicates a region to ignore)",
        required=False,
    )
    args = parser.parse_args()
    rusty_mws_processor(args.affinities_path, args.segmentation_path, args.mask_path)


def run_metrics():
    from funlib.persistence import open_ds
    from annotation_processing_utils.postprocess.metrics import (
        InstanceSegmentationOverlapAndScorer,
    )

    parser = argparse.ArgumentParser(
        description="Run instance segmentation metrics for a particular ground truth and test dataset"
    )
    parser.add_argument(
        "--gt_path", type=str, help="Path to the ground truth data", required=True
    )
    parser.add_argument(
        "--test_path", type=str, help="Path to the test data", required=True
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="Path to the mask data",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--metrics_path", type=str, help="Path for the output metrics", required=True
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of compute workers", required=True
    )

    args = parser.parse_args()

    gt_filename, gt_ds_name = parse_data_path(args.gt_path)
    gt_array = open_ds(gt_filename, gt_ds_name)

    test_filename, test_ds_name = parse_data_path(args.test_path)
    test_array = open_ds(test_filename, test_ds_name)

    if args.mask_path is None:
        mask_array = None
    else:
        mask_filename, mask_ds_name = parse_data_path(args.mask_path)
        mask_array = open_ds(mask_filename, mask_ds_name)

    username = getpass.getuser()
    timestamp = datetime.now().strftime(f"%Y%m%d/%H%M%S/")
    log_dir = f"/nrs/cellmap/{username}/logs/daisy_logs/metrics_logs/{test_ds_name}/{timestamp}"

    os.makedirs(log_dir, exist_ok=True)
    isos = InstanceSegmentationOverlapAndScorer(
        gt_array=gt_array,
        test_array=test_array,
        mask_array=mask_array,
        metrics_path=args.metrics_path,
        log_dir=log_dir,
        num_workers=args.num_workers,
    )
    isos.process()
